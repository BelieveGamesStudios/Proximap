import numpy as np
import pyrr

class Camera:
    def __init__(self):
        # Focus/Target point in world coordinates (Z-up)
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Yaw (rotation around Z axis) in radians
        self.yaw = 0.785  # ~45 degrees
        # Pitch (angle above XY plane) in radians
        self.pitch = 0.523  # ~30 degrees
        # Distance from target
        self.distance = 8.0
        # Projection settings
        self.is_perspective = True
        self.fov = 45.0  # Field of View in Y direction (degrees)
        self.ortho_scale = 6.0  # Height of the orthographic view area

    def snap_to_view(self, view: str, set_ortho=True):
        """Snaps the camera to canonical views (front, back, right, left, top, bottom)."""
        _SNAP_VIEWS = {
            "front":  (np.pi,       0.0),
            "back":   (0.0,         0.0),
            "right":  (np.pi/2.0,   0.0),
            "left":   (-np.pi/2.0,  0.0),
            "top":    (np.pi,       np.radians(89.0)),
            "bottom": (np.pi,      -np.radians(89.0)),
        }
        if view == "user":
            self.is_perspective = True
            return
        if view in _SNAP_VIEWS:
            yaw, pitch = _SNAP_VIEWS[view]
            self.yaw = yaw
            self.pitch = pitch
            if set_ortho:
                self.is_perspective = False

    def frame_object(self, obj):
        """Frames the camera on the selected object's bounding box (Unity 'F' shortcut)."""
        world_min, world_max = obj.get_world_aabb()
        center = (world_min + world_max) / 2.0
        extent = np.linalg.norm(world_max - world_min) * 0.5
        self.target = center
        self.distance = max(extent * 2.5, 1.0)
        if not self.is_perspective:
            self.ortho_scale = max(extent * 2.5, 1.0)


    def get_position(self):
        # Calculate eye position in Z-up system:
        # X = tx + r * cos(pitch) * sin(yaw)
        # Y = ty - r * cos(pitch) * cos(yaw)
        # Z = tz + r * sin(pitch)
        cos_pitch = np.cos(self.pitch)
        x = self.target[0] + self.distance * cos_pitch * np.sin(self.yaw)
        y = self.target[1] - self.distance * cos_pitch * np.cos(self.yaw)
        z = self.target[2] + self.distance * np.sin(self.pitch)
        return np.array([x, y, z], dtype=np.float32)

    def get_view_matrix(self):
        eye = self.get_position()
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        # Guard against looking straight up/down where cross product would fail
        if np.abs(np.cos(self.pitch)) < 1e-4:
            up = np.array([0.0, 1.0, 0.0], dtype=np.float32) if self.pitch > 0 else np.array([0.0, -1.0, 0.0], dtype=np.float32)
        return pyrr.matrix44.create_look_at(eye, self.target, up)

    def get_projection_matrix(self, aspect):
        if self.is_perspective:
            return pyrr.matrix44.create_perspective_projection(self.fov, aspect, 0.1, 100.0)
        else:
            half_h = self.ortho_scale * 0.5
            half_w = half_h * aspect
            return pyrr.matrix44.create_orthogonal_projection(-half_w, half_w, -half_h, half_h, 0.1, 100.0)

    def orbit(self, delta_yaw, delta_pitch):
        self.yaw += delta_yaw
        # Constrain pitch to avoid flipping over the poles (-89 to +89 degrees)
        max_pitch = np.radians(89.0)
        self.pitch = np.clip(self.pitch + delta_pitch, -max_pitch, max_pitch)

    def pan(self, delta_x, delta_y):
        # Pan relative to the camera's orientation
        # 1. Compute view direction and camera right/up vectors
        eye = self.get_position()
        forward = self.target - eye
        forward = forward / np.linalg.norm(forward)
        
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if np.abs(np.dot(forward, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            
        right = np.cross(forward, world_up)
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        up = up / np.linalg.norm(up)
        
        # Scale movement based on zoom distance (responsive panning speed)
        scale = self.distance * 0.004
        if not self.is_perspective:
            scale = self.ortho_scale * 0.004
            
        self.target += right * (-delta_x * scale) + up * (delta_y * scale)

    def get_forward_vector(self):
        """Returns normalized 3D forward direction vector in world coordinates (Z-up)."""
        cos_pitch = np.cos(self.pitch)
        fwd = np.array([
            -cos_pitch * np.sin(self.yaw),
            cos_pitch * np.cos(self.yaw),
            -np.sin(self.pitch)
        ], dtype=np.float32)
        norm = np.linalg.norm(fwd)
        return fwd / norm if norm > 1e-6 else np.array([0.0, 1.0, 0.0], dtype=np.float32)

    def get_right_vector(self):
        """Returns normalized 3D right direction vector in world coordinates (Z-up)."""
        fwd = self.get_forward_vector()
        world_up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        if np.abs(np.dot(fwd, world_up)) > 0.99:
            world_up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        right = np.cross(fwd, world_up)
        norm = np.linalg.norm(right)
        return right / norm if norm > 1e-6 else np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def get_up_vector(self):
        """Returns normalized 3D up direction vector in camera space."""
        fwd = self.get_forward_vector()
        right = self.get_right_vector()
        up = np.cross(right, fwd)
        norm = np.linalg.norm(up)
        return up / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0], dtype=np.float32)

    def move(self, delta_vector):
        """Translates the camera target (and eye) by a delta displacement in world coordinates."""
        self.target = self.target + np.array(delta_vector, dtype=np.float32)

    def fps_look(self, delta_yaw, delta_pitch):
        """Rotates camera orientation around the camera's eye position in place (Unity Fly-Look)."""
        current_eye = self.get_position()
        self.yaw += delta_yaw
        max_pitch = np.radians(89.0)
        self.pitch = np.clip(self.pitch + delta_pitch, -max_pitch, max_pitch)
        
        # Keep eye position fixed, update target based on new yaw/pitch
        cos_pitch = np.cos(self.pitch)
        offset = np.array([
            self.distance * cos_pitch * np.sin(self.yaw),
            -self.distance * cos_pitch * np.cos(self.yaw),
            self.distance * np.sin(self.pitch)
        ], dtype=np.float32)
        self.target = current_eye - offset

    def zoom(self, factor):
        if self.is_perspective:
            self.distance = max(self.distance - factor, 0.5)
        else:
            self.ortho_scale = max(self.ortho_scale - factor, 0.5)


class Mesh:
    """Represents a GL Mesh container with vertices, indices, normals, UVs, texture, and AABB."""
    def __init__(self, vertices, indices, normals, local_aabb_min, local_aabb_max, texcoords=None, texture_data=None):
        self.vertices = np.array(vertices, dtype=np.float32)
        self.indices = np.array(indices, dtype=np.uint32)
        self.normals = np.array(normals, dtype=np.float32)
        self.texcoords = np.array(texcoords, dtype=np.float32) if texcoords is not None else None
        
        # PIL Image or None
        self.texture_data = texture_data
        
        # Local space Axis-Aligned Bounding Box (AABB)
        self.local_aabb_min = np.array(local_aabb_min, dtype=np.float32)
        self.local_aabb_max = np.array(local_aabb_max, dtype=np.float32)
        
        # OpenGL buffer/texture handles
        self.vao = None
        self.vbo = None
        self.ebo = None
        self.vbo_normals = None
        self.vbo_texcoords = None
        self.texture_id = None

    def setup_buffers(self):
        # Deferred binding to be called with an active OpenGL context
        import OpenGL.GL as gl
        
        # Create and bind VAO
        self.vao = gl.glGenVertexArrays(1)
        gl.glBindVertexArray(self.vao)
        
        # VBO for positions (layout = 0)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.vertices.nbytes, self.vertices, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        
        # VBO for normals (layout = 1)
        self.vbo_normals = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_normals)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self.normals.nbytes, self.normals, gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 3, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
        
        # VBO for texture coordinates (layout = 2)
        if self.texcoords is not None and len(self.texcoords) > 0:
            self.vbo_texcoords = gl.glGenBuffers(1)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo_texcoords)
            gl.glBufferData(gl.GL_ARRAY_BUFFER, self.texcoords.nbytes, self.texcoords, gl.GL_STATIC_DRAW)
            gl.glEnableVertexAttribArray(2)
            gl.glVertexAttribPointer(2, 2, gl.GL_FLOAT, gl.GL_FALSE, 0, None)
            
        # EBO for indices
        self.ebo = gl.glGenBuffers(1)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self.ebo)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self.indices.nbytes, self.indices, gl.GL_STATIC_DRAW)
        
        gl.glBindVertexArray(0)
        
        # Upload texture to OpenGL
        if self.texture_data is not None:
            try:
                image = self.texture_data
                if image.mode not in ('RGB', 'RGBA'):
                    image = image.convert('RGBA')
                img_width, img_height = image.size
                
                # Flip vertically for OpenGL's bottom-left UV coordinate origin
                img_np = np.array(image)
                img_np = np.flipud(img_np)
                img_bytes = img_np.tobytes()
                
                img_format = gl.GL_RGBA if image.mode == 'RGBA' else gl.GL_RGB
                
                self.texture_id = gl.glGenTextures(1)
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
                
                # Setup texture filtering/wrapping
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_REPEAT)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_REPEAT)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR_MIPMAP_LINEAR)
                gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
                
                gl.glTexImage2D(
                    gl.GL_TEXTURE_2D, 0, img_format, img_width, img_height,
                    0, img_format, gl.GL_UNSIGNED_BYTE, img_bytes
                )
                gl.glGenerateMipmap(gl.GL_TEXTURE_2D)
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            except Exception as e:
                try:
                    print(f"[Mesh Setup GL Texture Error]: {type(e).__name__}")
                except Exception:
                    print("[Mesh Setup GL Texture Error]: (error could not be formatted)")
                self.texture_id = None

    def draw(self):
        import OpenGL.GL as gl
        if self.vao is not None:
            # Bind texture if available
            if self.texture_id is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
            gl.glBindVertexArray(self.vao)
            gl.glDrawElements(gl.GL_TRIANGLES, len(self.indices), gl.GL_UNSIGNED_INT, None)
            gl.glBindVertexArray(0)
            if self.texture_id is not None:
                gl.glBindTexture(gl.GL_TEXTURE_2D, 0)

    def cleanup(self):
        import OpenGL.GL as gl
        # Safely clean up buffers if context is alive
        try:
            if self.vao is not None:
                gl.glDeleteVertexArrays(1, [self.vao])
                gl.glDeleteBuffers(1, [self.vbo])
                gl.glDeleteBuffers(1, [self.vbo_normals])
                if self.vbo_texcoords is not None:
                    gl.glDeleteBuffers(1, [self.vbo_texcoords])
                gl.glDeleteBuffers(1, [self.ebo])
                if self.texture_id is not None:
                    gl.glDeleteTextures(1, [self.texture_id])
                self.vao = self.vbo = self.vbo_normals = self.ebo = None
        except Exception:
            pass


class Object:
    """A Node in the scene. Holds transform properties and a reference to a Mesh."""
    def __init__(self, name: str, mesh: Mesh):
        self.name = name
        self.mesh = mesh
        
        # Transform properties
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Rotation stored natively as Euler angles in degrees (X, Y, Z)
        self.rotation = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.scale = np.array([1.0, 1.0, 1.0], dtype=np.float32)

    def get_model_matrix(self) -> np.ndarray:
        import trimesh
        rx = np.radians(-self.rotation[0])
        ry = np.radians(-self.rotation[1])
        rz = np.radians(-self.rotation[2])
        
        t_mat = trimesh.transformations.translation_matrix(self.position)
        r_mat = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
        s_mat = np.diag([self.scale[0], self.scale[1], self.scale[2], 1.0])
        
        # Column-major transformation: T * R * S
        col_major = t_mat @ r_mat @ s_mat
        
        # Return row-major (transpose) for rendering/shader compatibility
        return col_major.T.astype(np.float32)

    def get_outline_model_matrix(self, scale_factor: float) -> np.ndarray:
        import trimesh
        rx = np.radians(-self.rotation[0])
        ry = np.radians(-self.rotation[1])
        rz = np.radians(-self.rotation[2])
        
        t_mat = trimesh.transformations.translation_matrix(self.position)
        r_mat = trimesh.transformations.euler_matrix(rx, ry, rz, 'sxyz')
        s_mat = np.diag([self.scale[0] * scale_factor, self.scale[1] * scale_factor, self.scale[2] * scale_factor, 1.0])
        
        col_major = t_mat @ r_mat @ s_mat
        return col_major.T.astype(np.float32)

    def get_world_aabb(self) -> tuple:
        """Computes the world-space bounding box by transforming the local AABB."""
        model_mat = self.get_model_matrix()
        local_min = self.mesh.local_aabb_min
        local_max = self.mesh.local_aabb_max
        
        # Compute all 8 corners of the local bounding box
        corners = [
            np.array([local_min[0], local_min[1], local_min[2], 1.0]),
            np.array([local_min[0], local_min[1], local_max[2], 1.0]),
            np.array([local_min[0], local_max[1], local_min[2], 1.0]),
            np.array([local_min[0], local_max[1], local_max[2], 1.0]),
            np.array([local_max[0], local_min[1], local_min[2], 1.0]),
            np.array([local_max[0], local_min[1], local_max[2], 1.0]),
            np.array([local_max[0], local_max[1], local_min[2], 1.0]),
            np.array([local_max[0], local_max[1], local_max[2], 1.0]),
        ]
        
        # Transform corners to world space
        world_corners = [pyrr.matrix44.apply_to_vector(model_mat, c)[:3] for c in corners]
        
        # Extract new min and max bounds from world corners
        world_corners = np.array(world_corners)
        world_min = np.min(world_corners, axis=0)
        world_max = np.max(world_corners, axis=0)
        return world_min, world_max


# Procedural geometry creators
def create_cube_mesh() -> Mesh:
    # 24 vertices (4 per face to support independent normals)
    vertices = [
        # Z+ (Top)
        -0.5, -0.5, 0.5,   0.5, -0.5, 0.5,   0.5, 0.5, 0.5,  -0.5, 0.5, 0.5,
        # Z- (Bottom)
        -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5, 0.5, -0.5, -0.5, 0.5, -0.5,
        # X+ (Right)
         0.5, -0.5, -0.5,  0.5,  0.5, -0.5,  0.5, 0.5,  0.5,  0.5, -0.5,  0.5,
        # X- (Left)
        -0.5, -0.5, -0.5, -0.5,  0.5, -0.5, -0.5, 0.5,  0.5, -0.5, -0.5,  0.5,
        # Y+ (Back)
        -0.5,  0.5, -0.5,  0.5,  0.5, -0.5,  0.5, 0.5,  0.5, -0.5, 0.5,  0.5,
        # Y- (Front)
        -0.5, -0.5, -0.5,  0.5, -0.5, -0.5,  0.5, -0.5, 0.5, -0.5, -0.5, 0.5
    ]
    
    normals = [
        # Z+
        0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   0.0, 0.0, 1.0,   0.0, 0.0, 1.0,
        # Z-
        0.0, 0.0, -1.0,  0.0, 0.0, -1.0,  0.0, 0.0, -1.0,  0.0, 0.0, -1.0,
        # X+
        1.0, 0.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 0.0,   1.0, 0.0, 0.0,
        # X-
        -1.0, 0.0, 0.0,  -1.0, 0.0, 0.0,  -1.0, 0.0, 0.0,  -1.0, 0.0, 0.0,
        # Y+
        0.0, 1.0, 0.0,   0.0, 1.0, 0.0,   0.0, 1.0, 0.0,   0.0, 1.0, 0.0,
        # Y-
        0.0, -1.0, 0.0,  0.0, -1.0, 0.0,  0.0, -1.0, 0.0,  0.0, -1.0, 0.0
    ]
    
    indices = [
        0, 1, 2,  2, 3, 0,        # Z+
        4, 6, 5,  6, 4, 7,        # Z- (flipped winding order for proper inside-out)
        8, 9, 10, 10, 11, 8,      # X+
        12, 14, 13, 14, 12, 15,   # X-
        16, 17, 18, 18, 19, 16,   # Y+
        20, 22, 21, 22, 20, 23    # Y-
    ]
    
    return Mesh(vertices, indices, normals, [-0.5, -0.5, -0.5], [0.5, 0.5, 0.5])


def create_plane_mesh() -> Mesh:
    # A single square plane in the XY flat space
    vertices = [
        -1.0, -1.0, 0.0,
         1.0, -1.0, 0.0,
         1.0,  1.0, 0.0,
        -1.0,  1.0, 0.0
    ]
    normals = [
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0,
        0.0, 0.0, 1.0
    ]
    indices = [
        0, 1, 2,
        2, 3, 0
    ]
    return Mesh(vertices, indices, normals, [-1.0, -1.0, 0.0], [1.0, 1.0, 0.0])


def create_sphere_mesh(rings=16, sectors=32) -> Mesh:
    vertices = []
    normals = []
    indices = []
    
    R = 1.0 / float(rings - 1)
    S = 1.0 / float(sectors - 1)
    
    for r in range(rings):
        for s in range(sectors):
            y = np.sin(-np.pi/2.0 + np.pi * r * R)
            x = np.cos(2.0 * np.pi * s * S) * np.sin(np.pi * r * R)
            z = np.sin(2.0 * np.pi * s * S) * np.sin(np.pi * r * R)
            
            # Since this is a Z-up, let's swap z and y for standard Z-up sphere
            # Z points up, X/Y form horizontal ground.
            vx, vy, vz = x, z, y
            
            vertices.extend([vx, vy, vz])
            normals.extend([vx, vy, vz])  # Unit sphere normals match vertex positions
            
    for r in range(rings - 1):
        for s in range(sectors - 1):
            i00 = r * sectors + s
            i01 = r * sectors + (s + 1)
            i10 = (r + 1) * sectors + s
            i11 = (r + 1) * sectors + (s + 1)
            
            indices.extend([i00, i01, i11])
            indices.extend([i11, i10, i00])
            
    return Mesh(vertices, indices, normals, [-1.0, -1.0, -1.0], [1.0, 1.0, 1.0])


class Scene:
    """Manages scene objects, selection state, and ray-AABB intersections."""
    def __init__(self):
        self.objects = []
        self.selected_objects = []
        self.active_object = None

    @property
    def selected_object(self):
        return self.active_object or (self.selected_objects[-1] if self.selected_objects else None)

    @selected_object.setter
    def selected_object(self, value):
        self.active_object = value
        if value is None:
            self.selected_objects = []
        elif value not in self.selected_objects:
            self.selected_objects = [value]

    def add_object(self, obj: Object):
        self.objects.append(obj)

    def remove_object(self, obj: Object):
        if obj in self.objects:
            self.objects.remove(obj)
            if obj in self.selected_objects:
                self.selected_objects.remove(obj)
            if self.active_object == obj:
                self.active_object = self.selected_objects[-1] if self.selected_objects else None

    def clear(self):
        self.objects.clear()
        self.selected_objects.clear()
        self.active_object = None

    def perform_picking(self, mouse_x: float, mouse_y: float, viewport_width: float, viewport_height: float, camera: Camera, shift_held: bool = False):
        """
        Calculates world-space ray and runs Ray-AABB Slab intersection test
        against all active objects. Selects the closest hit object.
        """
        # 1. Convert mouse coordinates to Normalized Device Coordinates (NDC)
        ndc_x = (2.0 * mouse_x) / viewport_width - 1.0
        ndc_y = 1.0 - (2.0 * mouse_y) / viewport_height

        # NDC ray starts at near plane and ends at far plane
        ray_ndc_near = np.array([ndc_x, ndc_y, -1.0, 1.0], dtype=np.float32)
        ray_ndc_far  = np.array([ndc_x, ndc_y,  1.0, 1.0], dtype=np.float32)

        # 2. Get matrices
        aspect   = viewport_width / max(viewport_height, 1.0)
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect)

        # pyrr uses row-vector convention: clip = v · V · P, so VP = V·P
        vp_mat     = pyrr.matrix44.multiply(view_mat, proj_mat)
        inv_vp_mat = pyrr.matrix44.inverse(vp_mat)

        # 3. Unproject ray endpoints to world coordinates
        world_near = pyrr.matrix44.apply_to_vector(inv_vp_mat, ray_ndc_near)
        world_far  = pyrr.matrix44.apply_to_vector(inv_vp_mat, ray_ndc_far)

        # Perspective divide to recover world-space positions
        world_near = world_near[:3] / world_near[3]
        world_far  = world_far[:3]  / world_far[3]

        ray_origin = world_near
        ray_dir    = world_far - world_near
        ray_len    = np.linalg.norm(ray_dir)
        if ray_len < 1e-12:
            return None
        ray_dir = ray_dir / ray_len

        # Camera forward vector (into the scene) for depth pre-rejection
        cam_pos     = camera.get_position()
        cam_forward = camera.target - cam_pos
        cam_fwd_len = np.linalg.norm(cam_forward)
        if cam_fwd_len > 1e-12:
            cam_forward = cam_forward / cam_fwd_len
        else:
            cam_forward = np.array([0.0, 1.0, 0.0], dtype=np.float32)

        closest_hit_obj = None
        min_t = float('inf')

        # 4. Ray-AABB intersection for each object
        for obj in self.objects:
            bounds_min, bounds_max = obj.get_world_aabb()

            # --- Screen-space pre-rejection: skip objects whose projected AABB
            #     is entirely outside the clicked NDC pixel column/row.
            #     Cheap 2-D test before the full 3-D slab test.
            corners_local = [
                np.array([bounds_min[0], bounds_min[1], bounds_min[2], 1.0]),
                np.array([bounds_min[0], bounds_min[1], bounds_max[2], 1.0]),
                np.array([bounds_min[0], bounds_max[1], bounds_min[2], 1.0]),
                np.array([bounds_min[0], bounds_max[1], bounds_max[2], 1.0]),
                np.array([bounds_max[0], bounds_min[1], bounds_min[2], 1.0]),
                np.array([bounds_max[0], bounds_min[1], bounds_max[2], 1.0]),
                np.array([bounds_max[0], bounds_max[1], bounds_min[2], 1.0]),
                np.array([bounds_max[0], bounds_max[1], bounds_max[2], 1.0]),
            ]
            ss_xs, ss_ys = [], []
            all_behind = True
            for c in corners_local:
                clip = np.dot(c, vp_mat)          # row-vector · VP
                if clip[3] > 0.001:               # in front of near plane
                    all_behind = False
                    ss_xs.append(clip[0] / clip[3])
                    ss_ys.append(clip[1] / clip[3])
                elif clip[3] <= 0.0:
                    pass  # behind camera; do not contribute
                else:
                    all_behind = False

            # All corners behind camera → definitely not visible, skip
            if all_behind:
                continue

            # If we gathered screen-space projections, do a quick 2-D overlap
            # check against the clicked NDC position with a small epsilon margin
            if ss_xs:
                margin = 0.05  # small tolerance in NDC units
                if (max(ss_xs) < ndc_x - margin or min(ss_xs) > ndc_x + margin or
                        max(ss_ys) < ndc_y - margin or min(ss_ys) > ndc_y + margin):
                    continue   # projected AABB doesn't overlap the click point

            # --- 3-D Slab Ray-AABB test ---
            hit, t_near, t_far = self.ray_aabb_intersect_full(ray_origin, ray_dir, bounds_min, bounds_max)
            if not hit:
                continue

            # Only accept hits where the *front face* of the AABB is in front of
            # the camera (t_near >= 0).  If the ray starts inside the box, t_near
            # is negative — use t_far but only if t_far is also positive (i.e. the
            # box straddles the ray origin, meaning the object truly surrounds the
            # camera eye, which is a legitimate closest-object pick).
            if t_near >= 0.0:
                t_hit = t_near
            elif t_far >= 0.0:
                # Ray origin is inside the AABB (camera inside the mesh bounds).
                # In practice this should be uncommon; use t_far so we still pick it
                # but give it a large bias so a proper front-face hit always wins.
                t_hit = t_far + 1e6
            else:
                continue  # entire AABB is behind the ray origin → skip

            if t_hit < min_t:
                min_t = t_hit
                closest_hit_obj = obj

        if closest_hit_obj is not None:
            if shift_held:
                if closest_hit_obj in self.selected_objects:
                    self.selected_objects.remove(closest_hit_obj)
                    if self.active_object == closest_hit_obj:
                        self.active_object = self.selected_objects[-1] if self.selected_objects else None
                else:
                    self.selected_objects.append(closest_hit_obj)
                    self.active_object = closest_hit_obj
            else:
                self.selected_objects = [closest_hit_obj]
                self.active_object = closest_hit_obj
        else:
            if not shift_held:
                self.selected_objects = []
                self.active_object = None

        return closest_hit_obj

    def perform_box_picking(self, x1: float, y1: float, x2: float, y2: float, viewport_width: float, viewport_height: float, camera: Camera, shift_held: bool = False):
        """
        Selects all objects that are inside or intersect the screen-space selection rectangle.
        If shift_held, toggle selections or add to selection.
        """
        box_x1 = min(x1, x2)
        box_x2 = max(x1, x2)
        box_y1 = min(y1, y2)
        box_y2 = max(y1, y2)

        aspect = viewport_width / max(viewport_height, 1.0)
        view_mat = camera.get_view_matrix()
        proj_mat = camera.get_projection_matrix(aspect)
        vp_mat = pyrr.matrix44.multiply(view_mat, proj_mat)

        newly_selected = []
        for obj in self.objects:
            local_min = obj.mesh.local_aabb_min
            local_max = obj.mesh.local_aabb_max
            corners = [
                np.array([local_min[0], local_min[1], local_min[2], 1.0]),
                np.array([local_min[0], local_min[1], local_max[2], 1.0]),
                np.array([local_min[0], local_max[1], local_min[2], 1.0]),
                np.array([local_min[0], local_max[1], local_max[2], 1.0]),
                np.array([local_max[0], local_min[1], local_min[2], 1.0]),
                np.array([local_max[0], local_min[1], local_max[2], 1.0]),
                np.array([local_max[0], local_max[1], local_min[2], 1.0]),
                np.array([local_max[0], local_max[1], local_max[2], 1.0]),
            ]
            model_mat = obj.get_model_matrix()
            world_corners = [pyrr.matrix44.apply_to_vector(model_mat, c) for c in corners]
            
            sx_list = []
            sy_list = []
            for wc in world_corners:
                clip_pos = pyrr.matrix44.apply_to_vector(vp_mat, wc)
                if clip_pos[3] > 0.0:
                    ndc = clip_pos[:3] / clip_pos[3]
                    sx = (ndc[0] + 1.0) * 0.5 * viewport_width
                    sy = (1.0 - ndc[1]) * 0.5 * viewport_height
                    sx_list.append(sx)
                    sy_list.append(sy)
            
            if sx_list and sy_list:
                min_sx = min(sx_list)
                max_sx = max(sx_list)
                min_sy = min(sy_list)
                max_sy = max(sy_list)
                
                overlap = not (max_sx < box_x1 or min_sx > box_x2 or max_sy < box_y1 or min_sy > box_y2)
                if overlap:
                    newly_selected.append(obj)
        
        if shift_held:
            for obj in newly_selected:
                if obj not in self.selected_objects:
                    self.selected_objects.append(obj)
                    self.active_object = obj
                else:
                    self.selected_objects.remove(obj)
                    if self.active_object == obj:
                        self.active_object = self.selected_objects[-1] if self.selected_objects else None
        else:
            self.selected_objects = newly_selected
            self.active_object = newly_selected[-1] if newly_selected else None

        return self.selected_objects

    @staticmethod
    def ray_aabb_intersect(origin, direction, box_min, box_max) -> tuple:
        """
        Slab method for Ray-AABB intersection.
        Returns (has_hit: bool, t_intersection: float)
        Legacy wrapper — prefer ray_aabb_intersect_full for picking.
        """
        hit, t_near, t_far = Scene.ray_aabb_intersect_full(origin, direction, box_min, box_max)
        if not hit:
            return False, 0.0
        t_hit = t_near if t_near >= 0.0 else t_far
        return True, t_hit

    @staticmethod
    def ray_aabb_intersect_full(origin, direction, box_min, box_max) -> tuple:
        """
        Slab method for Ray-AABB intersection.
        Returns (has_hit: bool, t_near: float, t_far: float).
        t_near is the entry distance and t_far is the exit distance along the ray.
        Both can be negative if the box is fully or partially behind the ray origin.
        The caller is responsible for deciding which t value to use.
        """
        t_min = -float('inf')
        t_max =  float('inf')

        for i in range(3):
            if np.abs(direction[i]) < 1e-8:
                # Ray is parallel to the slab; if origin is outside, no hit
                if origin[i] < box_min[i] or origin[i] > box_max[i]:
                    return False, 0.0, 0.0
            else:
                t1 = (box_min[i] - origin[i]) / direction[i]
                t2 = (box_max[i] - origin[i]) / direction[i]

                t_entry = min(t1, t2)
                t_exit  = max(t1, t2)

                t_min = max(t_min, t_entry)
                t_max = min(t_max, t_exit)

        if t_min <= t_max:
            return True, t_min, t_max

        return False, 0.0, 0.0


def apply_texture_to_meshes(meshes, texture_path):
    import os
    from PIL import Image

    if not texture_path or not os.path.exists(texture_path):
        return
    
    ext = os.path.splitext(texture_path)[1].lower()
    img_path = texture_path
    
    if ext == '.mtl':
        try:
            mtl_dir = os.path.dirname(texture_path)
            with open(texture_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    parts = line.strip().split()
                    if parts and parts[0] == 'map_Kd':
                        rel_img = line.strip()[6:].strip()
                        candidate_path = os.path.join(mtl_dir, rel_img)
                        if os.path.exists(candidate_path):
                            img_path = candidate_path
                            break
                        base_img = os.path.basename(rel_img)
                        candidate_path_base = os.path.join(mtl_dir, base_img)
                        if os.path.exists(candidate_path_base):
                            img_path = candidate_path_base
                            break
        except Exception as e:
            print(f"[Texture Import] Error parsing MTL file: {e}")
            
        if img_path == texture_path:
            return
            
    try:
        image = Image.open(img_path)
        image.load()
        
        for node_name, mesh in meshes:
            mesh.texture_data = image
            if mesh.texcoords is None or len(mesh.texcoords) == 0:
                verts = mesh.vertices.reshape(-1, 3)
                min_val = np.min(verts, axis=0) if len(verts) > 0 else np.array([0,0,0])
                max_val = np.max(verts, axis=0) if len(verts) > 0 else np.array([1,1,1])
                diff = max_val - min_val
                diff[diff == 0.0] = 1.0
                
                dims = np.argsort(diff)
                u_dim = dims[2]
                v_dim = dims[1]
                
                uvs = np.zeros((len(verts), 2), dtype=np.float32)
                uvs[:, 0] = (verts[:, u_dim] - min_val[u_dim]) / diff[u_dim]
                uvs[:, 1] = (verts[:, v_dim] - min_val[v_dim]) / diff[v_dim]
                mesh.texcoords = uvs.flatten()
    except Exception as e:
        print(f"[Texture Import] Error loading image: {e}")


def load_mesh_file(file_path: str) -> list[tuple[str, Mesh]]:
    """Loads a mesh file (.obj, .glb, .gltf, .ply) using trimesh and returns a list of (node_name, Mesh) tuples."""
    import trimesh
    mesh_data = trimesh.load(file_path)

    if isinstance(mesh_data, trimesh.Scene):
        if not mesh_data.geometry:
            raise ValueError("The loaded 3D scene contains no geometries.")
        # Collect (node_name, world-space_geometry) pairs by walking the scene graph
        named_geoms = []
        for node_name in mesh_data.graph.nodes_geometry:
            transform, geometry_name = mesh_data.graph.get(node_name)
            geom = mesh_data.geometry.get(geometry_name)
            if geom is None:
                continue
            # Apply the node's world transform to bake geometry into world space
            original_visual = geom.visual
            world_geom = geom.copy()
            world_geom.apply_transform(transform)
            if hasattr(original_visual, 'uv') and (not hasattr(world_geom.visual, 'uv') or world_geom.visual.uv is None):
                try:
                    world_geom.visual = original_visual
                except Exception:
                    pass
            named_geoms.append((node_name, world_geom))
    else:
        named_geoms = [(None, mesh_data)]  # single mesh, no node name

    results = []
    for node_name, geom in named_geoms:
        vertices = geom.vertices.astype(np.float32)
        indices = geom.faces.astype(np.uint32)

        # Extract normals, compute if missing
        if hasattr(geom, 'vertex_normals') and geom.vertex_normals is not None and len(geom.vertex_normals) > 0:
            normals = geom.vertex_normals.astype(np.float32)
        else:
            geom.vertex_normals = trimesh.geometry.weighted_vertex_normals(
                len(geom.vertices), geom.faces, geom.face_normals, geom.face_angles
            )
            normals = geom.vertex_normals.astype(np.float32)

        # Apply photogrammetry coordinate flip on load to normalize to Y-up
        from point_cloud_io import apply_photogrammetry_coordinate_flip
        vertices, normals, _, _ = apply_photogrammetry_coordinate_flip(points=vertices, normals=normals)

        # Extract texture coords and material texture image if present
        texcoords = None
        texture_data = None
        if hasattr(geom, 'visual') and geom.visual is not None:
            if hasattr(geom.visual, 'uv') and geom.visual.uv is not None and len(geom.visual.uv) > 0:
                raw_uv = geom.visual.uv
                n_verts = len(geom.vertices)
                n_face_verts = len(geom.faces) * 3
                
                if len(raw_uv) == n_verts:
                    texcoords = raw_uv.astype(np.float32)
                elif len(raw_uv) == n_face_verts:
                    # Face-indexed UVs: remap to vertex-indexed UVs
                    texcoords_remapped = np.zeros((n_verts, 2), dtype=np.float32)
                    for face_idx, face in enumerate(geom.faces):
                        for local_idx, vert_idx in enumerate(face):
                            texcoords_remapped[vert_idx] = raw_uv[face_idx * 3 + local_idx]
                    texcoords = texcoords_remapped
                else:
                    texcoords = raw_uv.astype(np.float32)
                
                # Extract material image
                if hasattr(geom.visual, 'material') and geom.visual.material is not None:
                    mat = geom.visual.material
                    if hasattr(mat, 'image') and mat.image is not None:
                        texture_data = mat.image
                    elif hasattr(mat, 'baseColorTexture') and mat.baseColorTexture is not None:
                        texture_data = mat.baseColorTexture

        local_min = np.min(vertices, axis=0).astype(np.float32) if len(vertices) > 0 else np.zeros(3, dtype=np.float32)
        local_max = np.max(vertices, axis=0).astype(np.float32) if len(vertices) > 0 else np.zeros(3, dtype=np.float32)

        mesh = Mesh(
            vertices.flatten(), 
            indices.flatten(), 
            normals.flatten(), 
            local_min, 
            local_max, 
            texcoords.flatten() if texcoords is not None else None, 
            texture_data
        )
        results.append((node_name, mesh))

    return results


def export_scene_to_file(scene, file_path: str, pipeline_native_coords: bool = True):
    """Parent all scene objects to a single world mesh and export it to OBJ, GLB, or USDZ.

    If pipeline_native_coords is True (default for pipeline export), applies the inverse matrix
    F^T = F (180 deg X-axis rotation) before file serialization so the exported mesh matches
    native COLMAP/OpenMVS Y-down coordinates and aligns with scene.mvs camera poses.
    
    If saving for external tools like Blender, pass pipeline_native_coords=False.
    """
    import trimesh
    import numpy as np
    from point_cloud_io import apply_photogrammetry_coordinate_flip
    
    if not scene.objects:
        raise ValueError("No objects in the scene to export.")
        
    export_scene = trimesh.Scene()
    
    for obj in scene.objects:
        vertices = obj.mesh.vertices.reshape(-1, 3)
        faces = obj.mesh.indices.reshape(-1, 3)
        
        # Transform the vertices to world space using the object's model matrix
        model_mat = obj.get_model_matrix()
        world_vertices = []
        for v in vertices:
            v4 = np.array([v[0], v[1], v[2], 1.0], dtype=np.float32)
            wv = pyrr.matrix44.apply_to_vector(model_mat, v4)[:3]
            world_vertices.append(wv)
        world_vertices = np.array(world_vertices, dtype=np.float32)
        
        world_normals = np.zeros((0, 3), dtype=np.float32)
        if obj.mesh.normals is not None and len(obj.mesh.normals) > 0:
            normals = obj.mesh.normals.reshape(-1, 3)
            try:
                inv_trans_model = np.linalg.inv(model_mat).T
                wn_list = []
                for n in normals:
                    n4 = np.array([n[0], n[1], n[2], 0.0], dtype=np.float32)
                    wn = pyrr.matrix44.apply_to_vector(inv_trans_model, n4)[:3]
                    length = np.linalg.norm(wn)
                    if length > 0.0:
                        wn /= length
                    wn_list.append(wn)
                world_normals = np.array(wn_list, dtype=np.float32)
            except Exception:
                pass
                
        if pipeline_native_coords:
            # Revert to native COLMAP/OpenMVS coordinates (F^T = F)
            world_vertices, world_normals, _, _ = apply_photogrammetry_coordinate_flip(
                points=world_vertices, normals=world_normals
            )

        # Create trimesh for this object
        t_mesh = trimesh.Trimesh(vertices=world_vertices, faces=faces, process=False)
        if world_normals is not None and len(world_normals) > 0:
            t_mesh.vertex_normals = world_normals

        # Handle UVs and texture mapping
        if obj.mesh.texcoords is not None and len(obj.mesh.texcoords) > 0 and obj.mesh.texture_data is not None:
            t_mesh.visual = trimesh.visual.TextureVisuals(
                uv=obj.mesh.texcoords.reshape(-1, 2),
                material=trimesh.visual.material.SimpleMaterial(image=obj.mesh.texture_data)
            )
            
        export_scene.add_geometry(t_mesh, node_name=obj.name)
        
    # Export to chosen format (Scene supports glb, gltf, obj)
    if file_path.lower().endswith(".usdz"):
        merged = trimesh.util.concatenate(list(export_scene.geometry.values()))
        _export_usdz_from_trimesh(merged, file_path)
    else:
        export_scene.export(file_path)


def _export_usdz_from_trimesh(mesh, file_path: str):
    """
    Export a trimesh Trimesh object to a .usdz file using a pure-Python writer.
    USDZ is a ZIP64 archive containing a USDA (ASCII USD) text file.
    No openusd/pxr installation required.
    """
    import zipfile
    import io
    import numpy as np

    # Handle both Trimesh and Scene instances (if concatenated failed or passed directly)
    if hasattr(mesh, "geometry") and len(mesh.geometry) > 0:
        import trimesh
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.int32)
    
    # Format USD arrays
    pts = ", ".join(f"({v[0]:.6f}, {v[1]:.6f}, {v[2]:.6f})" for v in vertices)
    cnts = ", ".join(["3"] * len(faces))
    idxs = ", ".join(str(i) for f in faces for i in f)
    
    usda = f"""#usda 1.0
(
    defaultPrim = "Root"
    metersPerUnit = 1
    upAxis = "Y"
)

def Xform "Root"
{{
    def Mesh "Mesh"
    {{
        int[] faceVertexCounts = [{cnts}]
        int[] faceVertexIndices = [{idxs}]
        point3f[] points = [{pts}]
        uniform token subdivisionScheme = "none"
    }}
}}
"""
    # USDZ specification requires uncompressed files aligned properly inside the zip file
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        zf.writestr("model.usda", usda)
    with open(file_path, "wb") as f:
        f.write(buf.getvalue())


