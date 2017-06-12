#version 150
in vec3 position;
out vec4 v_position;
uniform mat4 perspective;
uniform mat4 modelview;
void main() {
  v_position = modelview * vec4(position, 1.0);
  gl_Position = perspective * v_position;
}
