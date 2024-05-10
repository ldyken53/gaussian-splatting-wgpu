@group(0) @binding(0) var render_target : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<uniform> canvas_size: vec2<u32>;

// the workgroup size needs to be the tile size
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > canvas_size.x || global_id.y > canvas_size.y) {
        return;
    }
    textureStore(render_target, global_id.xy, vec4<f32>(1.0, 1.0, 0.0, 1.0f));
}