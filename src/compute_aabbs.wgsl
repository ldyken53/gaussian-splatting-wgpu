struct PointInput {
    @location(0) position: vec3<f32>,
    @location(1) value: f32,
    @location(2) log_scale: vec3<f32>,
    @location(3) opacity: f32,
    @location(4) rot: vec4<f32>,
};
struct AABBs {
    start_cell: vec3<u32>,
    end_cell: vec3<u32>,
    value: f32
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
        let R : mat3x3<f32> = build_rotation(gaussian.rot);
        // Bound Gaussians by 3 * the standard deviation
        let m = 1.0;
        let S : vec3<f32> = vec3<f32>(
          exp(gaussian.log_scale.x) * m, 
          exp(gaussian.log_scale.y) * m, 
          exp(gaussian.log_scale.z) * m,
        );
        let n = array<f32, 2>(-1.0, 1.0);
        var mins = gaussian.position;
        var maxes = gaussian.position;
        for (var i = 0; i < 2; i++) {
          for (var j = 0; j < 2; j++) {
            for (var k = 0; k < 2; k++) {
              let corner = gaussian.position + n[i] * R[0] * S[0] + n[j] * R[1] * S[1] + n[k] * R[2] * S[2];
              maxes = max(maxes, corner);
              mins = min(mins, corner);
            }
          }
        }
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