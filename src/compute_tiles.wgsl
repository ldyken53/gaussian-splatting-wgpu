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
@group(0) @binding(4) var<storage, read> point_data: array<vec4<f32>>;
@group(0) @binding(5) var<storage, read> tetra_uvs: array<mat4x2<f32>>;
@group(0) @binding(6) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(7) var<uniform> tile_size: u32;
@group(0) @binding(8) var<uniform> uniforms: Uniforms;

// the workgroup size needs to be the tile size
@compute @workgroup_size(TILE_SIZE_MACRO, TILE_SIZE_MACRO)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id: vec3<u32>, @builtin(workgroup_id) w_id: vec3<u32>, @builtin(num_workgroups) n_wgs: vec3<u32>) {
    if (global_id.x > canvas_size.x || global_id.y > canvas_size.y) {
        return;
    }
    let face_indices = array<vec4<u32>, 3>(
        vec4<u32>(0, 0, 0, 1),
        vec4<u32>(1, 3, 2, 3),
        vec4<u32>(2, 1, 3, 2),
    );
    let pixel = vec2<f32>(global_id.xy);
    let tile_id = w_id.x + w_id.y * n_wgs.x;
    var start_index : u32 = 0;
    if (tile_id > 0) {
        start_index = ranges[tile_id - 1];
    }
    var end_index : u32 = ranges[tile_id];
    var accumulated_color: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);  
    let float_canvas = vec2(f32(canvas_size.x), f32(canvas_size.y));
    var done = false;
    for (var i = start_index; i < end_index; i++) {
        let tetra_id : u32 = u32(indices[i] / 4);
        let tetra = tetra_data[tetra_id];
        let uvs = tetra_uvs[tetra_id];
        let face_id = indices[i] % 4;
        if (is_point_in_triangle(
            pixel, 
            uvs[face_indices[0][face_id]] * float_canvas, 
            uvs[face_indices[1][face_id]] * float_canvas, 
            uvs[face_indices[2][face_id]] * float_canvas
        )) {
            // let value = trilinear_interpolate(
            //     pixel, 
            //     uvs[face_indices[0][face_id]] * float_canvas, 
            //     uvs[face_indices[1][face_id]] * float_canvas, 
            //     uvs[face_indices[2][face_id]] * float_canvas,
            //     point_data[tetra[face_indices[0][face_id]]].w, 
            //     point_data[tetra[face_indices[1][face_id]]].w, 
            //     point_data[tetra[face_indices[2][face_id]]].w
            // );
            let value = (
                point_data[tetra[face_indices[0][face_id]]].w + 
                point_data[tetra[face_indices[1][face_id]]].w + 
                point_data[tetra[face_indices[2][face_id]]].w
            ) / 3.0;
            // value = (value + 13.0) / 26.0;
            accumulated_color = vec3<f32>(value / 5, 0.0, 1.0);
            done = true;
            break;
        }
        if (done) {
            break;
        }
    }
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

fn trilinear_interpolate(
    p: vec2<f32>,
    v1: vec2<f32>,
    v2: vec2<f32>,
    v3: vec2<f32>,
    f1: f32,
    f2: f32,
    f3: f32
) -> f32 {
    let v12 = v2 - v1;
    let v13 = v3 - v1;
    let u = dot(v12, p - v1) / dot(v12, v12);
    let v = dot(v13, p - v1) / dot(v13, v13);
    let w = 1.0 - u - v;
    return f1 * w + f2 * u + f3 * v;
}