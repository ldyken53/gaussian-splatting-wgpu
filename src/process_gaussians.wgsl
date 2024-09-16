struct PointInput {
    @location(0) position: vec3<f32>,
    @location(1) value: f32,
    @location(2) log_scale: vec3<f32>,
    @location(3) rot: vec4<f32>,
};
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

@group(0) @binding(0) var<storage, read> point_data: array<PointInput>;
@group(0) @binding(1) var<storage, read_write> gaussian_data: array<GaussianData>;
@group(0) @binding(2) var<storage, read_write> tile_counts: array<u32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;
@group(0) @binding(4) var<uniform> n_unpadded: u32;
@group(0) @binding(5) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(6) var<uniform> tile_size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= n_unpadded) {
        tile_counts[global_id.x] = 0;
        return;
    } else {
        let gaussian = point_data[global_id.x];
        let pos = vec4<f32>(gaussian.position, 1.0f);

        // exit if gaussian outside frustum (depth < 0.2 or x, y < -1 or x, y > 1)
        if (!in_frustum(pos)) {
            tile_counts[global_id.x] = 0;
            return;
        }

        let view_pos = uniforms.view_matrix * pos;
        var proj_pos = uniforms.proj_matrix * pos;
        let inv_w = 1.0 / (proj_pos.w + 0.0000001f);
        proj_pos = proj_pos * inv_w;
        let point_uv = (proj_pos.xy * 0.5) + 0.5;

        // compute covariance and conics
        let cov_2d = compute_cov2d(gaussian.position, gaussian.log_scale, gaussian.rot);
        let det = cov_2d.x * cov_2d.z - cov_2d.y * cov_2d.y;
        // TODO: test if this is needed
        if (det == 0.0f) {
            tile_counts[global_id.x] = 0;
            return;
        }        
        let det_inv = 1.0 / det;
        let conic = vec3<f32>(
            cov_2d.z * det_inv, 
            -cov_2d.y * det_inv, 
            cov_2d.x * det_inv
        );
        // Compute extent in screen space (by finding eigenvalues of
        // 2D covariance matrix). Use extent to compute a bounding rectangle
        // of screen-space tiles that this Gaussian overlaps with. Quit if
        // rectangle covers 0 tiles. 
        let mid = 0.5 * (cov_2d.x + cov_2d.z);
        let lambda_1 = mid + sqrt(max(0.1, mid * mid - det));
        let lambda_2 = mid - sqrt(max(0.1, mid * mid - det));
        let radius = ceil(3. * sqrt(max(lambda_1, lambda_2)));

        let num_tiles = vec2<f32>(ceil(f32(canvas_size.x) / f32(tile_size)), ceil(f32(canvas_size.y) / f32(tile_size)));
        let rect = getRect(
            point_uv,
            radius,
            canvas_size,
            num_tiles
        );
        tile_counts[global_id.x] = (rect.w - rect.y) * (rect.z - rect.x);

        // need to use tile id for more significant bits, and rounded depth for least significant for proper ordering

        // writing tile_id just for the gaussian mean position
        // let tile_id = u32(point_uv.x * num_tiles.x) + u32(floor(point_uv.y * num_tiles.y) * num_tiles.x);
        // tiles[global_id.x] = tile_id * 1000 + u32(depth);

        let color = vec3<f32>((gaussian.value - 0.1926) / (4.9775 - 0.1926), 0.0, 1.0);
        let opacity = 1.0;
        // save data so it doesn't have to be recomputed when computing tiles
        gaussian_data[global_id.x] = GaussianData(
            point_uv,
            conic,
            view_pos.z,
            color,
            opacity,
            rect
        );
    }
}

fn in_frustum(world_pos: vec4<f32>) -> bool {
  let p_hom = uniforms.proj_matrix * world_pos;
  let p_w = 1.0f / (p_hom.w + 0.0000001f);

  let p_proj = vec3(
    p_hom.x * p_w,
    p_hom.y * p_w,
    p_hom.z * p_w
  );

  let p_view = uniforms.view_matrix * world_pos;
  
  if (p_view.z <= 0.2f || ((p_proj.x <= -1.1 || p_proj.x >= 1.1 || p_proj.y <= -1.1 || p_proj.y >= 1.1))) {
    return false;
  }

  return true;
}

fn compute_cov3d(log_scale: vec3<f32>, rot: vec4<f32>) -> array<f32, 6> {

  let modifier = 1.0;

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

fn sigmoid(x: f32) -> f32 {
  // if (x >= 0.) {
  //   return 1. / (1. + exp(-x));
  // } else {
  //   let z = exp(x);
  //   return z / (1. + z);
  // }

  let z = exp(x);
  let condition = f32(x >= 0.);

  return (condition * (1. / (1. + exp(-x)))) + ((1.0 - condition) * (z / (1. + z)));
}

// TODO: Fix issue with last tile being full of intersecitons
fn getRect(uv: vec2<f32>, max_radius: f32, grid: vec2<u32>, num_tiles: vec2<f32>) -> vec4<u32> {
  let p: vec2<f32> = uv * vec2(f32(grid.x), f32(grid.y));
  let t_s = i32(tile_size);

  var rect_min: vec2<u32>;
  var rect_max: vec2<u32>;

  // Calculate rect_min
  let min_x = min(i32(num_tiles.x), max(0, i32(p.x - max_radius) / t_s));
  rect_min.x = u32(min_x);

  let min_y = min(i32(num_tiles.y), max(0, i32(p.y - max_radius) / t_s));
  rect_min.y = u32(min_y);

  // Calculate rect_max
  let max_x = min(i32(num_tiles.x), max(0, i32(p.x + max_radius) / t_s)) + 1;
  rect_max.x = u32(max_x);

  let max_y = min(i32(num_tiles.y), max(0, i32(p.y + max_radius)/ t_s)) + 1;
  rect_max.y = u32(i32(max_y));

  return vec4<u32>(rect_min, rect_max);
}