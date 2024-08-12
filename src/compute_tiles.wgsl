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
@group(0) @binding(1) var<storage, read> ranges: array<u32>;
@group(0) @binding(2) var<storage, read> indices: array<u32>;
@group(0) @binding(3) var<storage, read> tetra_data: array<vec4<u32>>;
@group(0) @binding(4) var<storage, read_write> tetra_uvs: array<mat4x2<f32>>;
@group(0) @binding(5) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(6) var<uniform> tile_size: u32;
@group(0) @binding(7) var<uniform> uniforms: Uniforms;

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
    for (var i = start_index; i < end_index; i++) {
        let tetra = tetra_data[indices[i]];
        let uvs = tetra_uvs[indices[i]];
        if (is_point_in_quad(pixel, uvs[0], uvs[1], uvs[2], uvs[3])) {
            if (indices[i] == 0) {
                accumulated_color = vec3<f32>(0.0, 0.0, 1.0);
            } else if (indices[i] == 1) {
                accumulated_color = vec3<f32>(0.0, 0.0, 1.0);
            } else {
                accumulated_color = vec3<f32>(0.0, 0.0, 1.0);
            }
            break;
        }
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

fn cross2D(a: vec2<f32>, b: vec2<f32>, c: vec2<f32>) -> f32 {
    let ab = a - b;
    let cb = c - b;
    return ab.x * cb.y - ab.y * cb.x;
}

fn is_point_in_triangle(p: vec2<f32>, v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>) -> bool {
    let cross1 = cross2D(v0, v1, p);
    let cross2 = cross2D(v1, v2, p);
    let cross3 = cross2D(v2, v0, p);

    // If all cross products have the same sign, the point is inside the triangle
    let all_positive = cross1 > 0.0 && cross2 > 0.0 && cross3 > 0.0;
    let all_negative = cross1 < 0.0 && cross2 < 0.0 && cross3 < 0.0;

    return all_positive || all_negative;
}

fn is_point_in_quad(p: vec2<f32>, v0: vec2<f32>, v1: vec2<f32>, v2: vec2<f32>, v3: vec2<f32>) -> bool {
    // calculate cross products for each pair of consecutive edges
    let cross1 = cross2D(v0, v1, v2);
    let cross2 = cross2D(v1, v2, v3);
    let cross3 = cross2D(v2, v3, v0);
    let cross4 = cross2D(v3, v0, v1);

    // check if all cross products have the same sign
    let all_positive = cross1 > 0.0 && cross2 > 0.0 && cross3 > 0.0 && cross4 > 0.0;
    let all_negative = cross1 < 0.0 && cross2 < 0.0 && cross3 < 0.0 && cross4 < 0.0;
    if (all_positive || all_negative) {
        let cross1 = cross2D(v0, v1, p);
        let cross2 = cross2D(v1, v2, p);
        let cross3 = cross2D(v2, v3, p);
        let cross4 = cross2D(v3, v0, p);

        let all_positive = cross1 > 0.0 && cross2 > 0.0 && cross3 > 0.0 && cross4 > 0.0;
        let all_negative = cross1 < 0.0 && cross2 < 0.0 && cross3 < 0.0 && cross4 < 0.0;

        return all_positive || all_negative;
    } else {
        if sign(cross1) != sign(cross2) && sign(cross1) != sign(cross3) && sign(cross1) != sign(cross4) {
            return is_point_in_triangle(p, v0, v2, v3); // v1 is inside the triangle formed by v0, v2, and v3
        } else if sign(cross2) != sign(cross3) && sign(cross2) != sign(cross4) && sign(cross2) != sign(cross1) {
            return is_point_in_triangle(p, v0, v1, v3); // v2 is inside the triangle formed by v0, v1, and v3
        } else if sign(cross3) != sign(cross4) && sign(cross3) != sign(cross1) && sign(cross3) != sign(cross2) {
            return is_point_in_triangle(p, v0, v1, v2); // v3 is inside the triangle formed by v0, v1, and v2
        } else {
            return is_point_in_triangle(p, v1, v2, v3);; // v0 is inside the triangle formed by v1, v2, and v3
        }
    }
}