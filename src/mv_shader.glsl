#version 150
in vec3 position;
in vec3 normal;
in vec2 tex_coords;
out vec3 v_normal;
out vec4 v_position;
out vec2 v_tex_coords;
uniform mat4 perspective;
uniform mat4 modelview;
uniform mat4 normal_mat;
void main() {
  v_normal = mat3(normal_mat) * normal;
  v_position = modelview * vec4(position, 1.0);
  v_tex_coords = tex_coords;
  gl_Position = perspective * v_position;
}
