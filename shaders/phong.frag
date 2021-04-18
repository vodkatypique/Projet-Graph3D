#version 330 core

#define NB_OF_LIGHTS 2

// fragment position and normal of the fragment, in WORLD coordinates
in vec3 w_position, w_normal;

// light dir, in world coordinates
uniform vec3 light_dir[NB_OF_LIGHTS];
uniform vec3 light_pos[NB_OF_LIGHTS];
uniform vec3 k_s;

// material properties
uniform vec3 k_d, k_a;
uniform float s;

// world camera position
uniform vec3 w_camera_position;

out vec4 out_color;

void main() {
    // Compute all vectors, oriented outwards from the fragment
    vec3 n = normalize(w_normal);
    vec3 v = normalize(w_camera_position - w_position);

    vec3 diffuse_color = vec3(0, 0, 0);
    vec3 specular_color = vec3(0, 0, 0);

    for(int i = 0; i < NB_OF_LIGHTS; i++){

        vec3 l = normalize(-light_dir[i]);
        vec3 r = reflect(-l, n);

        vec3 p = (w_position - light_pos[i]) / 500;
        float d_k = dot(p, p);

        diffuse_color += k_d * max(dot(n, l), 0) / d_k;
        specular_color += k_s * pow(max(dot(r, v), 0), s) / d_k;
    }

    out_color =  vec4(k_a, 1) + vec4(diffuse_color, 1) + vec4(specular_color, 1);
}