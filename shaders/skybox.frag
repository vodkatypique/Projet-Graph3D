
#version 330 core

uniform samplerCube diffuse_map;
in vec3 frag_tex_coords;
out vec4 out_color;

void main() {
    out_color = texture(diffuse_map, frag_tex_coords);
}
