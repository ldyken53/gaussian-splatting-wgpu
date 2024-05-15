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

@group(0) @binding(0) var<storage, read_write> vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> depths: array<f32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> tiles: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;
@group(0) @binding(5) var<uniform> n_unpadded: u32;
@group(0) @binding(6) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(7) var<uniform> tile_size: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    indices[global_id.x] = global_id.x;
    if (global_id.x >= n_unpadded) {
        depths[global_id.x] = 1e20f; // pad with +inf
        tiles[global_id.x] = 4294967294u;
    } else {
        let pos = vec4<f32>(vertices[global_id.x], 1.0f);
        if (!in_frustum(pos)) {
            depths[global_id.x] = 1e20f; // pad with +inf
            tiles[global_id.x] = 4294967294u;
            return;
        }
        let proj_pos = uniforms.view_matrix * pos;
        // just for debugging if needed
        depths[global_id.x] = proj_pos.z;
        if (proj_pos.x < 0 || proj_pos.x >= 1 || proj_pos.y < 0 || proj_pos.y >= 1) {
            tiles[global_id.x] = 4294967294u;
        } else {
            let t = tile_size;
            let num_tiles = vec2<f32>(ceil(f32(canvas_size.x) / f32(tile_size)), ceil(f32(canvas_size.y) / f32(tile_size)));
            let tile_id = u32(proj_pos.x * num_tiles.x) + u32(floor(proj_pos.y * num_tiles.y) * num_tiles.x);
            // need to use tile id for more significant bits, and rounded depth for least significant for proper ordering
            // TODO: Fix when this overflows for large number of tiles
            // TODO: negative depths are clipped completely
            // TODO: maybe need to divide by proj_pos.w?
            var depth = proj_pos.z;
            if (depth < 0) {
                tiles[global_id.x] = 4294967294u;
            } else {
                // TODO: assumes depths are 0-9.99
                depth = min(depth * 100, 999);
                tiles[global_id.x] = tile_id * 1000 + u32(depth);
            }
        }
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
  
  if (p_view.z <= 0.2f || ((p_proj.x < -1.1 || p_proj.x > 1.1 || p_proj.y < -1.1 || p_proj.y > 1.1))) {
    return false;
  }

  return true;
}