@group(0) @binding(0) var render_target : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> ranges: array<u32>;
@group(0) @binding(2) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(3) var<uniform> tile_size: u32;

// the workgroup size needs to be the tile size
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) w_id: vec3<u32>, @builtin(num_workgroups) n_wgs: vec3<u32>) {
    if (global_id.x > canvas_size.x || global_id.y > canvas_size.y) {
        return;
    }
    let tile_id = w_id.x + w_id.y * n_wgs.x;
    var num_gaussians : f32 = 0;
    if (tile_id == 0) {
        num_gaussians = f32(ranges[0]);
    } else {
        num_gaussians = f32(ranges[tile_id] - ranges[tile_id - 1]);
    }
    var color: vec2<f32> = vec2<f32>(num_gaussians / 50);
    // if ((w_id.x + w_id.y) % 2 == 0) {
    //     color = vec2<f32>(0.0, 0.0);
    // }
    let color2 = vec2<f32>(w_id.xy) / vec2<f32>(canvas_size / tile_size);
    textureStore(render_target, global_id.xy, vec4<f32>(color.x, color.y, 0.0, 1.0f));
}