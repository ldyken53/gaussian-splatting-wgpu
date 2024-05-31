// Draw a full screen quad using two triangles
struct VertexOutput {
  @builtin(position) position : vec4<f32>,
};
const pos : array<vec4<f32>, 6> = array<vec4<f32>, 6>(
	vec4<f32>(-1, 1, 0.5, 1),
	vec4<f32>(-1, -1, 0.5, 1),
	vec4<f32>(1, 1, 0.5, 1),
	vec4<f32>(-1, -1, 0.5, 1),
	vec4<f32>(1, 1, 0.5, 1),
	vec4<f32>(1, -1, 0.5, 1)
);

@vertex
fn vertex_main(@builtin(vertex_index) vertex_index : u32)
     -> VertexOutput {
    var output : VertexOutput;
    output.position = pos[vertex_index];
    return output;
}


@group(0) @binding(0) var output_texture : texture_2d<f32>;
@group(0) @binding(1) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(2) var u_sampler : sampler;

@fragment
fn fragment_main(@builtin(position) frag_coord : vec4<f32>) -> @location(0) vec4<f32> {
    let color = textureSample(output_texture, u_sampler, vec2<f32>(frag_coord.x / f32(canvas_size.x), (frag_coord.y / f32(canvas_size.y))));
    return color;
}

