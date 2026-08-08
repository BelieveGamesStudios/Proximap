#version 330 core
out vec4 FragColor;
 
in vec3 fragPos;
in vec3 fragNormal;
in vec2 fragTexCoord;
 
uniform vec3 lightDir;         // Normalized direction of light (in world space)
uniform vec3 eyePos;           // Camera position for specular calculation
uniform vec3 objectColor;      // Base color of the object (fallback)
uniform vec4 overrideColor;    // Color to use for selection highlighting/wireframe overlays
uniform bool useOverrideColor; // Flags whether to override lighting and base color
uniform sampler2D textureSampler;
uniform bool useTexture;
uniform int uLightingMode;     // 0 = Flat (Unlit), 1 = Flux (Blinn-Phong Preview), 2 = Prism (PBR Stub)

void main() {
    if (useOverrideColor) {
        FragColor = overrideColor;
        return;
    }
    
    vec3 baseColor = useTexture ? texture(textureSampler, fragTexCoord).rgb : objectColor;

    if (uLightingMode == 0) {
        // Mode 0: Flat (Unlit passthrough)
        FragColor = vec4(baseColor, 1.0);
        return;
    }

    vec3 normal = normalize(fragNormal);
    vec3 lDir = normalize(lightDir);

    if (uLightingMode == 1) {
        // Mode 1: Flux (Analytic Blinn-Phong Preview)
        float diff = abs(dot(normal, lDir));
        vec3 viewDir = normalize(eyePos - fragPos);
        vec3 halfDir = normalize(lDir + viewDir);
        float spec = pow(max(dot(normal, halfDir), 0.0), 32.0);

        vec3 ambient = 0.30 * baseColor;
        vec3 diffuse = 0.60 * diff * baseColor;
        vec3 specular = 0.20 * spec * vec3(1.0);

        FragColor = vec4(ambient + diffuse + specular, 1.0);
        return;
    }

    if (uLightingMode == 2) {
        // Mode 2: Prism (PBR / LuxCore Stub - Fallback to Flux Preview for v1)
        // Extension point for v2 RPManager Cook-Torrance shader integration.
        float diff = abs(dot(normal, lDir));
        vec3 viewDir = normalize(eyePos - fragPos);
        vec3 halfDir = normalize(lDir + viewDir);
        float spec = pow(max(dot(normal, halfDir), 0.0), 64.0);

        vec3 ambient = 0.25 * baseColor;
        vec3 diffuse = 0.65 * diff * baseColor;
        vec3 specular = 0.30 * spec * vec3(1.0);

        FragColor = vec4(ambient + diffuse + specular, 1.0);
        return;
    }

    // Default Fallback
    FragColor = vec4(baseColor, 1.0);
}
