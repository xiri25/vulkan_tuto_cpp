#version 450

layout(binding = 0) uniform UniformBufferObject {
    mat4 model;
    mat4 view;
    mat4 proj;
} ubo;

layout(location = 0) in vec3 inPosition; // Input position
layout(location = 1) in vec3 inColor;    // Input color
layout(location = 2) in vec2 inTexCoord;    // Input texture

layout(location = 0) out vec3 fragColor;  // Output color to fragment shader
layout(location = 1) out vec2 fragTexCoord;  // Output color to fragment shader

void main() {
    gl_Position = ubo.proj * ubo.view * ubo.model * vec4(inPosition, 1.0); // Set the position
    fragColor = inColor; // Pass color to fragment shader
    fragTexCoord = inTexCoord;
}
