struct PointInput {
    @location(0) position: vec3<f32>,
    @location(1) value: f32,
    @location(2) log_scale: vec3<f32>,
    @location(3) opacity: f32,
    @location(4) rot: vec4<f32>,
};
struct AABBs {
    conic: array<f32, 6>,
    start_cell: vec3<u32>,
    det: f32,
    end_cell: vec3<u32>,
    value: f32,
    mean: vec3<f32>
};
struct Uniforms {
    volume_mins: vec3<f32>,
    cell_size: f32,
    volume_maxes: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> point_data: array<PointInput>;
@group(0) @binding(1) var<storage, read_write> aabbs: array<AABBs>;
@group(0) @binding(2) var<storage, read_write> cell_counts: array<u32>;
@group(0) @binding(3) var<uniform> n_unpadded: u32;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= n_unpadded) {
        cell_counts[global_id.x] = 0;
        return;
    } else {
        let gaussian = point_data[global_id.x];
        aabbs[global_id.x].mean = gaussian.position;
        let R : mat3x3<f32> = build_rotation(gaussian.rot);

        // Build scale matrix
        let S = mat3x3<f32>(
          exp(gaussian.log_scale.x), 0., 0.,
          0., exp(gaussian.log_scale.y), 0.,
          0., 0., exp(gaussian.log_scale.z),
        );
        
        // Compute 3D covariance matrix and precompute conic
        let M = S * R;
        let Sigma = transpose(M) * M;
        // Epsilon for numerical stability
        let epsilon = max(max(abs(Sigma[0][0]), abs(Sigma[1][1])), abs(Sigma[2][2])) * 1e-5;
        let cov3d = array<f32, 6> (
          Sigma[0][0] + epsilon,
          Sigma[0][1],
          Sigma[0][2],
          Sigma[1][1] + epsilon,
          Sigma[1][2],
          Sigma[2][2] + epsilon,
        );
        let conic = compute_3d_conic(cov3d);
        aabbs[global_id.x].conic = conic;

        // For normalization
        let a = cov3d[0]; // Sigma[0][0]
        let b = cov3d[1]; // Sigma[0][1]
        let c = cov3d[2]; // Sigma[0][2]
        let d = cov3d[3]; // Sigma[1][1]
        let e = cov3d[4]; // Sigma[1][2]
        let f = cov3d[5]; // Sigma[2][2]
        let det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
        aabbs[global_id.x].det = det;

        // Bound Gaussians by 3 * the standard deviation
        let m = 1.0;
        let scaled_S : vec3<f32> = vec3<f32>(
          S[0][0] * m, 
          S[1][1] * m, 
          S[2][2] * m,
        );

        // Find max and min (x,y,z) of bounding box
        let n = array<f32, 2>(-1.0, 1.0);
        var mins = gaussian.position;
        var maxes = gaussian.position;
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 2; k++) {
              let corner = gaussian.position + n[i] * R[0] * scaled_S[0] + n[j] * R[1] * scaled_S[1] + n[k] * R[2] * scaled_S[2];
              maxes = max(maxes, corner);
              mins = min(mins, corner);
            }
          }
        }
        
        // Compute max and min (x, y, z) of cells
        let start_cell = max(
            vec3<u32>(floor((mins - vec3<f32>(uniforms.volume_mins)) / uniforms.cell_size)),
            vec3<u32>(0)
        );
        let end_cell = min(
            vec3<u32>(ceil((maxes - vec3<f32>(uniforms.volume_mins)) / uniforms.cell_size)),
            vec3<u32>(ceil((uniforms.volume_maxes - uniforms.volume_mins) / uniforms.cell_size))
        );
        let cell_dims = end_cell - start_cell;
        cell_counts[global_id.x] = cell_dims.x * cell_dims.y * cell_dims.z;
        aabbs[global_id.x].start_cell = start_cell;
        aabbs[global_id.x].end_cell = end_cell;
        aabbs[global_id.x].value = gaussian.value;
    }
}

fn build_rotation(rot: vec4<f32>) -> mat3x3<f32> {
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

  return R;
}

fn compute_3d_conic(cov: array<f32, 6>) -> array<f32, 6> {    
    let a = cov[0]; // Sigma[0][0]
    let b = cov[1]; // Sigma[0][1]
    let c = cov[2]; // Sigma[0][2]
    let d = cov[3]; // Sigma[1][1]
    let e = cov[4]; // Sigma[1][2]
    let f = cov[5]; // Sigma[2][2]
    
    let det = a * (d * f - e * e) - b * (b * f - c * e) + c * (b * e - c * d);
    let det_inv = 1.0 / det;

    let conic_a = (d * f - e * e) * det_inv;
    let conic_b = (c * e - b * f) * det_inv;
    let conic_c = (b * e - c * d) * det_inv;
    let conic_d = (a * f - c * c) * det_inv;
    let conic_e = (b * c - a * e) * det_inv;
    let conic_f = (a * d - b * b) * det_inv;
    
    return array<f32, 6> (
      conic_a,
      conic_b,
      conic_c,
      conic_d,
      conic_e,
      conic_f,
    );
}