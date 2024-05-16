struct GaussianData {
    uv: vec2<f32>,
    conic: vec3<f32>,
    color: vec3<f32>,
    opacity: f32,
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
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(workgroup_id) w_id: vec3<u32>, @builtin(num_workgroups) n_wgs: vec3<u32>) {
    if (global_id.x > canvas_size.x || global_id.y > canvas_size.y) {
        return;
    }
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
    textureStore(render_target, global_id.xy, vec4<f32>(accumulated_color.x, accumulated_color.y, accumulated_color.z, 1.0f));
    // just to keep uniforms active for now
    let color2 = vec2<f32>(w_id.xy) / vec2<f32>(canvas_size / tile_size);
    let v = uniforms.view_matrix;
}

fn compute_cov3d(log_scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {

  let modifier = uniforms.scale_modifier;

  let S = mat3x3<f32>(
    exp(log_scale.x) * modifier, 0., 0.,
    0., exp(log_scale.y) * modifier, 0.,
    0., 0., exp(log_scale.z) * modifier,
  );
  
  // Normalize quaternion to get valid rotation
  // let quat = rot;
  let quat = rot / length(rot);
  let r = quat.x;
  let x = quat.y;
  let y = quat.z;
  let z = quat.w;

  let R = mat3x3(
    1. - 2. * (y * y + z * z), 2. * (x * y - r * z), 2. * (x * z + r * y),
    2. * (x * y + r * z), 1. - 2. * (x * x + z * z), 2. * (y * z - r * x),
    2. * (x * z - r * y), 2. * (y * z + r * x), 1. - 2. * (x * x + y * y),
  );

  let M = S * R;
  let Sigma = transpose(M) * M;

  return array<f32, 6> (
    Sigma[0][0],
    Sigma[0][1],
    Sigma[0][2],
    Sigma[1][1],
    Sigma[1][2],
    Sigma[2][2],
  );
}


fn compute_cov2d(position: vec3<f32>, log_scale: vec3<f32>, rot: vec4<f32>) -> vec3<f32> {
  let cov3d = compute_cov3d(log_scale, rot);

  // The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.

  var t = uniforms.view_matrix * vec4<f32>(position, 1.0);
  // let focal_x = 1.0;
  // let focal_y = 1.0;
  let focal_x = uniforms.focal_x;
  let focal_y = uniforms.focal_y;

  // Orig
  let limx = 1.3 * uniforms.tan_fovx;
  let limy = 1.3 * uniforms.tan_fovy;
  let txtz = t.x / t.z;
  let tytz = t.y / t.z;

  t.x = min(limx, max(-limx, txtz)) * t.z;
  t.y = min(limy, max(-limy, tytz)) * t.z;

  let J = mat3x3(
    focal_x / t.z,  0., -(focal_x * t.x) / (t.z * t.z),
    0., focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
    0., 0., 0.,
  );

  // this includes the transpose
  let W = mat3x3(
    uniforms.view_matrix[0][0], uniforms.view_matrix[1][0], uniforms.view_matrix[2][0],
    uniforms.view_matrix[0][1], uniforms.view_matrix[1][1], uniforms.view_matrix[2][1],
    uniforms.view_matrix[0][2], uniforms.view_matrix[1][2], uniforms.view_matrix[2][2],
  );

  let T = W * J;

  let Vrk = mat3x3(
    cov3d[0], cov3d[1], cov3d[2],
    cov3d[1], cov3d[3], cov3d[4],
    cov3d[2], cov3d[4], cov3d[5],
  );

  var cov = transpose(T) * transpose(Vrk) * T;

  // Apply low-pass filter: every Gaussian should be at least
  // one pixel wide/high. Discard 3rd row and column.
  cov[0][0] += 0.3;
  cov[1][1] += 0.3;


  return vec3(cov[0][0], cov[0][1], cov[1][1]);
}