#version 330 core
out vec4 FragColor;
 
in vec3 fragPos;
in vec3 fragNormal;
in vec2 fragTexCoord;
 
uniform vec3 lightDir;         // Normalized direction of light (in world space)
uniform vec3 objectColor;      // Base color of the object (fallback)
uniform vec4 overrideColor;    // Color to use for selection highlighting/wireframe overlays
uniform bool useOverrideColor; // Flags whether to override lighting and base color
uniform sampler2D textureSampler;
uniform bool useTexture;
 
void main() {
    if (useOverrideColor) {
        FragColor = overrideColor;
        return;
    }
    
    vec3 normal = normalize(fragNormal);
    vec3 lDir = normalize(lightDir);
    
    // Choose base color (textured or flat)
    vec3 baseColor = useTexture ? texture(textureSampler, fragTexCoord).rgb : objectColor;
    
    // Double-sided directional lighting (headlight style)
    float diff = abs(dot(normal, lDir));
    
    // Ambient + Diffuse color
    vec3 ambient = 0.25 * baseColor;
    vec3 diffuse = 0.65 * diff * baseColor;
    
    FragColor = vec4(ambient + diffuse, 1.0);
}
