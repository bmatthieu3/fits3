// shader.vert
#version 440

layout(location=0) in vec2 a_ndc;

layout(location=0) out vec2 ndc;

layout(set = 0, binding = 3)
uniform Window {
    vec4 size;
};

void main() {
    gl_Position = vec4(a_ndc.xy, 0.0, 1.0);
    ndc = vec2(a_ndc.x, a_ndc.y * size.y / size.x);
}