#version 330 core
out vec4 FragColor;

in vec3 nearPoint;
in vec3 farPoint;

uniform mat4 view;
uniform mat4 proj;

uniform vec4 gridColor1;       // Primary grid line color
uniform vec4 gridColor2;       // Secondary grid line color
uniform vec4 axisColorX;       // X axis highlight color (normally Red)
uniform vec4 axisColorY;       // Y axis highlight color (normally Green)
uniform float gridFade;        // Maximum visible/fade distance
uniform float gridThickness;   // Grid line thickness (in pixels)
uniform float subdivisions;    // Number of secondary divisions (e.g. 10.0)

// Helper to draw anti-aliased procedural grid lines
float getGrid(vec2 coord, float scale, float width) {
    vec2 grid_coord = coord * scale;
    // Calculate derivatives for constant screen-space thickness
    vec2 derivative = fwidth(grid_coord);
    vec2 grid_lines = abs(fract(grid_coord - 0.5) - 0.5) / (derivative * width);
    float line_min = min(grid_lines.x, grid_lines.y);
    return 1.0 - min(line_min, 1.0);
}

void main() {
    float dz = farPoint.z - nearPoint.z;
    if (abs(dz) < 1e-7) {
        discard;
    }
    
    // Intersect ray (nearPoint -> farPoint) with horizontal plane Z=0
    float t = -nearPoint.z / dz;
    if (t < 0.0) {
        discard; // Plane is behind the camera near plane
    }
    
    vec3 fragPos3D = nearPoint + t * (farPoint - nearPoint);
    
    // Write accurate depth for standard depth buffer comparisons (so meshes occlude the grid)
    vec4 clipSpacePos = proj * view * vec4(fragPos3D, 1.0);
    if (clipSpacePos.w <= 0.0) {
        discard;
    }
    float depth = clipSpacePos.z / clipSpacePos.w;
    // Convert NDC depth [-1, 1] to window space depth [0, 1] and clamp
    gl_FragDepth = clamp((depth + 1.0) * 0.5, 0.0, 1.0);
    
    vec2 coord = fragPos3D.xy;
    
    // 1. Draw primary grid lines (scale 1.0)
    float g1 = getGrid(coord, 1.0, gridThickness);
    
    // 2. Draw secondary grid lines (scale = subdivisions, e.g. 10x finer)
    float g2 = getGrid(coord, subdivisions, gridThickness * 0.7);
    
    // 3. Draw Red/Green Axis highlights at X=0 and Y=0
    // X-axis is line Y = 0
    float axis_y_width = gridThickness * 1.8;
    float dist_y = abs(fragPos3D.y) / (fwidth(fragPos3D.y) * axis_y_width);
    float axisX = 1.0 - min(dist_y, 1.0);
    
    // Y-axis is line X = 0
    float axis_x_width = gridThickness * 1.8;
    float dist_x = abs(fragPos3D.x) / (fwidth(fragPos3D.x) * axis_x_width);
    float axisY = 1.0 - min(dist_x, 1.0);
    
    // 4. Combine grid lines
    vec4 color = vec4(0.0);
    if (g2 > 0.0) {
        color = mix(color, gridColor2, g2 * 0.35);
    }
    if (g1 > 0.0) {
        color = mix(color, gridColor1, g1 * 0.75);
    }
    
    // Overlay axis lines
    if (axisX > 0.0) {
        color = mix(color, axisColorX, axisX);
    }
    if (axisY > 0.0) {
        color = mix(color, axisColorY, axisY);
    }
    
    // 5. Calculate fade factor based on camera distance (to avoid noise at horizon)
    vec4 viewSpacePos = view * vec4(fragPos3D, 1.0);
    float viewDist = length(viewSpacePos.xyz);
    float fade = 1.0 - smoothstep(gridFade * 0.4, gridFade, viewDist);
    
    if (color.a == 0.0 || fade <= 0.0) {
        discard;
    }
    
    color.a *= fade;
    FragColor = color;
}
