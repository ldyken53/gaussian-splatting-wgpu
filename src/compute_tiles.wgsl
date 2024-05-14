struct PointInput {
    @location(0) position: vec3<f32>,
    @location(1) log_scale: vec3<f32>,
    @location(2) rot: vec4<f32>,
    @location(3) opacity_logit: f32,
    sh: array<vec3<f32>, 16>,
};

struct Uniforms {
    viewMatrix: mat4x4<f32>,
    projMatrix: mat4x4<f32>,
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
@group(0) @binding(3) var<storage, read> point_data: array<PointInput>;
@group(0) @binding(4) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(5) var<uniform> tile_size: u32;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;

// spherical harmonic coefficients
const SH_C0 = 0.28209479177387814f;
const SH_C1 = 0.4886025119029199f;
const SH_C2 = array(
    1.0925484305920792f,
    -1.0925484305920792f,
    0.31539156525252005f,
    -1.0925484305920792f,
    0.5462742152960396f
);
const SH_C3 = array(
    -0.5900435899266435f,
    2.890611442640554f,
    -0.4570457994644658f,
    0.3731763325901154f,
    -0.4570457994644658f,
    1.445305721320277f,
    -0.5900435899266435f
);

fn compute_color_from_sh(position: vec3<f32>, sh: array<vec3<f32>, 16>) -> vec3<f32> {
    let dir = normalize(position - uniforms.camera_position);
    var result = SH_C0 * sh[0];

    // // if deg > 0
    // let x = dir.x;
    // let y = dir.y;
    // let z = dir.z;

    // result = result + SH_C1 * (-y * sh[1] + z * sh[2] - x * sh[3]);

    // let xx = x * x;
    // let yy = y * y;
    // let zz = z * z;
    // let xy = x * y;
    // let xz = x * z;
    // let yz = y * z;

    // // if (sh_degree > 1) {
    // result = result +
    //     SH_C2[0] * xy * sh[4] +
    //     SH_C2[1] * yz * sh[5] +
    //     SH_C2[2] * (2. * zz - xx - yy) * sh[6] +
    //     SH_C2[3] * xz * sh[7] +
    //     SH_C2[4] * (xx - yy) * sh[8];
    
    // // if (sh_degree > 2) {
    // result = result +
    //     SH_C3[0] * y * (3. * xx - yy) * sh[9] +
    //     SH_C3[1] * xy * z * sh[10] +
    //     SH_C3[2] * y * (4. * zz - xx - yy) * sh[11] +
    //     SH_C3[3] * z * (2. * zz - 3. * xx - 3. * yy) * sh[12] +
    //     SH_C3[4] * x * (4. * zz - xx - yy) * sh[13] +
    //     SH_C3[5] * z * (xx - yy) * sh[14] +
    //     SH_C3[6] * x * (xx - 3. * yy) * sh[15];

    // unconditional
    result = result + 0.5;

    return max(result, vec3<f32>(0.));
}

// the workgroup size needs to be the tile size
@compute @workgroup_size(4, 4)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) w_id: vec3<u32>, @builtin(num_workgroups) n_wgs: vec3<u32>) {
    if (global_id.x > canvas_size.x || global_id.y > canvas_size.y) {
        return;
    }
    let tile_id = w_id.x + w_id.y * n_wgs.x;
    // var num_gaussians : f32 = 0;
    // if (tile_id == 0) {
    //     num_gaussians = f32(ranges[0]);
    // } else {
    //     num_gaussians = f32(ranges[tile_id] - ranges[tile_id - 1]);
    // }
    // var color: vec2<f32> = vec2<f32>(num_gaussians / 50);
    var start_index : u32 = 0;
    if (tile_id > 0) {
        start_index = ranges[tile_id - 1];
    }
    var end_index : u32 = ranges[tile_id];
    var color: vec3<f32> = vec3<f32>(0.0);
    for (var i = start_index; i < end_index; i++) {
        color = compute_color_from_sh(point_data[indices[i]].position, point_data[indices[i]].sh);
    }
    let color2 = vec2<f32>(w_id.xy) / vec2<f32>(canvas_size / tile_size);
    textureStore(render_target, global_id.xy, vec4<f32>(color.x, color.y, color.z, 1.0f));
}