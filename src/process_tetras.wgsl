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

@group(0) @binding(0) var<storage, read> point_data: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read_write> tetra_data: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tile_counts: array<u32>;
@group(0) @binding(3) var<storage, read_write> face_rects: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> face_depths: array<f32>;
@group(0) @binding(5) var<storage, read_write> tetra_uvs: array<mat4x2<f32>>;
@group(0) @binding(6) var<uniform> uniforms: Uniforms;
@group(0) @binding(7) var<uniform> num_tetra: u32;
@group(0) @binding(8) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(9) var<uniform> tile_size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x >= num_tetra) {
        tile_counts[global_id.x * 4] = 0;
        tile_counts[global_id.x * 4 + 1] = 0;
        tile_counts[global_id.x * 4 + 2] = 0;
        tile_counts[global_id.x * 4 + 3] = 0;
        return;
    } else {
        let tetra = tetra_data[global_id.x];
        let tetra_pos = mat4x4<f32>(
          vec4<f32>(point_data[tetra.x].xyz, 1.0f),
          vec4<f32>(point_data[tetra.y].xyz, 1.0f),
          vec4<f32>(point_data[tetra.z].xyz, 1.0f),
          vec4<f32>(point_data[tetra.w].xyz, 1.0f)
        );

        // exit if any point of the tetra outside frustum (depth < 0.2 or x, y < -1 or x, y > 1)
        if (!in_frustum(tetra_pos[0]) || !in_frustum(tetra_pos[1]) || !in_frustum(tetra_pos[2]) || !in_frustum(tetra_pos[3])) {
            tile_counts[global_id.x * 4] = 0;
            tile_counts[global_id.x * 4 + 1] = 0;
            tile_counts[global_id.x * 4 + 2] = 0;
            tile_counts[global_id.x * 4 + 3] = 0;
            return;
        }

        let view_pos = mat4x4<f32>(
          uniforms.view_matrix * tetra_pos[0],
          uniforms.view_matrix * tetra_pos[1],
          uniforms.view_matrix * tetra_pos[2],
          uniforms.view_matrix * tetra_pos[3]
        );
        let proj_posf = mat4x4<f32>(
          uniforms.proj_matrix * tetra_pos[0],
          uniforms.proj_matrix * tetra_pos[1],
          uniforms.proj_matrix * tetra_pos[2],
          uniforms.proj_matrix * tetra_pos[3]
        );
        let float_canvas = vec2(f32(canvas_size.x), f32(canvas_size.y));
        let points_uv = mat4x2<f32>(
          ((proj_posf[0].xy * (1.0 / (proj_posf[0].w + 0.0000001f))) * 0.5 + 0.5),
          ((proj_posf[1].xy * (1.0 / (proj_posf[1].w + 0.0000001f))) * 0.5 + 0.5),
          ((proj_posf[2].xy * (1.0 / (proj_posf[2].w + 0.0000001f))) * 0.5 + 0.5),
          ((proj_posf[3].xy * (1.0 / (proj_posf[3].w + 0.0000001f))) * 0.5 + 0.5)
        );
        tetra_uvs[global_id.x] = points_uv;

        // correct order of vertices for face visibility checking
        let indices = array<vec4<u32>, 3>(
            vec4<u32>(0, 0, 0, 1),
            vec4<u32>(1, 3, 2, 3),
            vec4<u32>(2, 1, 3, 2),
        );
        let num_tiles = vec2<f32>(ceil(f32(canvas_size.x) / f32(tile_size)), ceil(f32(canvas_size.y) / f32(tile_size)));

        for (var i : u32 = 0; i < 4; i++) {
          // precompute which faces of the tetra face the camera
          if (is_face_visible(points_uv[indices[0][i]], points_uv[indices[1][i]], points_uv[indices[2][i]]) > 0) {
            // TODO: find better way to do intersection check
            var mins = float_canvas;
            var maxes = vec2<f32>(0.0, 0.0);
            var closest : f32 = 1000;
            for (var j = 0; j < 3; j++) {
              mins.x = min(mins.x, points_uv[indices[j][i]].x * float_canvas.x);
              maxes.x = max(maxes.x, points_uv[indices[j][i]].x * float_canvas.x);
              mins.y = min(mins.y, points_uv[indices[j][i]].y * float_canvas.y);
              maxes.y = max(maxes.y, points_uv[indices[j][i]].y * float_canvas.y);
              closest = min(closest, view_pos[indices[j][i]].z);
            }

            let rect = getRect(
                mins,
                maxes,
                num_tiles
            );
            tile_counts[global_id.x * 4 + i] = (rect.w - rect.y) * (rect.z - rect.x);
            face_rects[global_id.x * 4 + i] = rect;
            face_depths[global_id.x * 4 + i] = closest;
          } else {
            tile_counts[global_id.x * 4 + i] = 0;
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
  
  if (p_view.z <= 0.2f || ((p_proj.x <= -1 || p_proj.x >= 1 || p_proj.y <= -1 || p_proj.y >= 1))) {
    return false;
  }

  return true;
}

// TODO: Fix issue with last tile being full of intersecitons
fn getRect(mins: vec2<f32>, maxes: vec2<f32>, num_tiles: vec2<f32>) -> vec4<u32> {
  let t_s = i32(tile_size);

  var rect_min: vec2<u32>;
  var rect_max: vec2<u32>;

  // Calculate rect_min
  let min_x = min(i32(num_tiles.x), max(0, i32(mins.x) / t_s));
  rect_min.x = u32(min_x);

  let min_y = min(i32(num_tiles.y), max(0, i32(mins.y) / t_s));
  rect_min.y = u32(min_y);

  // Calculate rect_max
  let max_x = min(i32(num_tiles.x), max(0, i32(maxes.x) / t_s)) + 1;
  rect_max.x = u32(max_x);

  let max_y = min(i32(num_tiles.y), max(0, i32(maxes.y)/ t_s)) + 1;
  rect_max.y = u32(i32(max_y));

  return vec4<u32>(rect_min, rect_max);
}

fn is_face_visible(p1: vec2<f32>, p2: vec2<f32>, p3: vec2<f32>) -> f32 {
    return ((p2.x - p1.x) * (p3.y - p1.y)) - ((p2.y - p1.y) * (p3.x - p1.x));
}