#version 150
in vec3 v_normal;
in vec4 v_position;
in vec2 v_tex_coords;
out vec4 color;
uniform vec3 u_light;
uniform vec3 ambient_l;
uniform vec3 diffuse_c;
uniform vec3 specular_c;
uniform sampler2D tex;
void main() {
  vec3 position = v_position.xyz / v_position.w;
  vec3 light_dir = normalize(u_light - position);
  vec3 tcolor = vec3(texture(tex, v_tex_coords));

  float diffuse = max(0.0,dot(normalize(v_normal), light_dir));

  vec3 half_direction = normalize(light_dir - normalize(position));
  float specular = pow(max(dot(half_direction, normalize(v_normal)), 0.0), 32.0);

  color = vec4(
    ambient_l * diffuse_c * tcolor +
    diffuse * diffuse_c * tcolor +
    specular * specular_c * tcolor, 1.0
  );
}
