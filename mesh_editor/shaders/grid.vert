#version 330 core
layout (location = 0) in vec3 aPos;

uniform mat4 invView;
uniform mat4 invProj;

out vec3 nearPoint;
out vec3 farPoint;

vec3 UnprojectPoint(float x, float y, float z, mat4 invVP) {
    vec4 unprojectedPoint = invVP * vec4(x, y, z, 1.0);
    return unprojectedPoint.xyz / unprojectedPoint.w;
}

void main() {
    mat4 invVP = invView * invProj;
    nearPoint = UnprojectPoint(aPos.x, aPos.y, -1.0, invVP);
    farPoint = UnprojectPoint(aPos.x, aPos.y, 1.0, invVP);
    gl_Position = vec4(aPos, 1.0);
}
