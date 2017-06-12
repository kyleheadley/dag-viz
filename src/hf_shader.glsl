#version 150
in vec4 v_position;
out vec4 color;
uniform uint highlights;
uniform float time;
uniform vec3 direction;
const uint highlight = uint(1);
const uint source = uint(2);
const uint sink = uint(4);

//pereferences
const vec4 color_highlight = vec4(0.2,0.2,1.0,0.4);
const vec4 color_source = vec4(0.2,1.0,0.2,0.4);
const vec4 color_sink = vec4(1.0,0.2,0.2,0.4);
const vec4 color_both = vec4(1.0,1.0,0.2,0.4);
const float move_speed = 25.0;
const float width = 15.0;
const float balance = 0.3;

void main() {
  //vec3 direction = vec3(0.0,1.0,0.0);
  //vec3 position = v_position.xyz / v_position.w;
  vec2 screen_dir = vec2(direction.x,direction.y);

  color = vec4(0.0,0.0,0.0,0.0);
  if ((highlights & highlight) > uint(0)) {
    color = color_highlight;
  } else if ((highlights & (source | sink)) == (source | sink)) {
    color = color_both;
  } else if ((highlights & source) > uint(0)) {
    vec2 dist = time*move_speed*screen_dir;
    vec2 pos = gl_FragCoord.xy - dist;
    if (fract(floor(length(pos))/width) <= balance) {
      color = color_source;
    }
  } else if ((highlights & sink) > uint(0)) {
    vec2 dist = time*move_speed*screen_dir;
    vec2 pos = gl_FragCoord.xy - dist;
    if (fract(floor(length(pos))/width) <= balance) {
      color = color_sink;
    }
  }
}
