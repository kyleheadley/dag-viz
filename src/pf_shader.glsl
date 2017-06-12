#version 150
uniform uvec4 u_id;
out vec4 id;
void main() {
	id = vec4(u_id)/255.0;
}
