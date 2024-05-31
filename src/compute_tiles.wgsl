struct GaussianData {
    uv: vec2<f32>,
    conic: vec3<f32>,
    depth: f32,
    color: vec3<f32>,
    opacity: f32,
    rect: vec4<u32>,
}

struct Uniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    camera_position: vec3<f32>,
    tan_fovx: f32,
    tan_fovy: f32,
    focal_x: f32,
    focal_y: f32,
    scale_modifier: f32,
};

@group(0) @binding(0) var render_target : texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(1) var<storage, read_write> ranges: array<u32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<storage, read> gaussian_data: array<GaussianData>;
@group(0) @binding(4) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(5) var<uniform> tile_size: u32;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;

// the workgroup size needs to be the tile size
@compute @workgroup_size(TILE_SIZE_MACRO, TILE_SIZE_MACRO)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) w_id: vec3<u32>, @builtin(num_workgroups) n_wgs: vec3<u32>) {
    if (global_id.x > canvas_size.x || global_id.y > canvas_size.y) {
        return;
    }
    // for coloring borders of tiles for debugging
    // if (local_id.x == TILE_SIZE_MACRO - 1 || local_id.y == TILE_SIZE_MACRO - 1) {
    //     textureStore(render_target, global_id.xy, vec4<f32>(1.0, 0.0, 0.0, 1.0f));
    //     return;
    // }
    let pixel = vec2<f32>(global_id.xy);
    let tile_id = w_id.x + w_id.y * n_wgs.x;
    var start_index : u32 = 0;
    if (tile_id > 0) {
        start_index = ranges[tile_id - 1];
    }
    var end_index : u32 = ranges[tile_id];
    var accumulated_color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);  
    var t_i: f32 = 1.0; // The initial value of accumulated alpha (initial value of accumulated multiplication)
    for (var i = start_index; i < end_index; i++) {
        let gaussian = gaussian_data[indices[i]];
        let conic = gaussian.conic;
        let g_xy = vec2<f32>(gaussian.uv.x * f32(canvas_size.x), gaussian.uv.y * f32(canvas_size.y));
        let distance = vec2<f32>(
            g_xy.x - pixel.x,
            g_xy.y - pixel.y
        );
        let power = -0.5 *
            (conic.x * distance.x * distance.x + conic.z * distance.y * distance.y) -
            conic.y * distance.x * distance.y;
        let alpha = min(0.99, gaussian.opacity * exp(power));
        let test_t = t_i * (1.0 - alpha);
        // TODO: not checking for index condition
        let condition = f32(power <= 0.0f && alpha >= 1.0 / 255.0 && test_t >= 0.0001f);
        accumulated_color += condition * gaussian.color * alpha * t_i;
        t_i = condition * test_t + (1.0 - condition) * t_i;
    }
    // accumulated_color = vec3<f32>(f32(end_index - start_index) / 1000);
    // accumulated_color = vec3<f32>(f32(global_id.x) / f32(canvas_size.x), f32(global_id.y) / f32(canvas_size.y), 0);
    // let num_tiles = vec2<f32>(ceil(f32(canvas_size.x) / f32(tile_size)), ceil(f32(canvas_size.y) / f32(tile_size)));
    // accumulated_color = vec3<f32>(f32(end_index - start_index) / 100, f32(end_index - start_index) / 100, 0.0);
    textureStore(render_target, global_id.xy, vec4<f32>(accumulated_color.x, accumulated_color.y, accumulated_color.z, 1.0f));
    // just to keep uniforms active for now
    let color2 = vec2<f32>(w_id.xy) / vec2<f32>(canvas_size / tile_size);
    let v = uniforms.view_matrix;
}