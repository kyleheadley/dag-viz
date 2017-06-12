#version 150
in vec3 position;
uniform mat4 perspective;
uniform mat4 modelview;
void main() {
  vec4 v_position = modelview * vec4(position, 1.0);
  gl_Position = perspective * v_position;
}
