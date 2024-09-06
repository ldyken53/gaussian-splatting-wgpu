@group(0) @binding(0) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> face_rects: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> face_depths: array<vec3<f32>>;
@group(0) @binding(3) var<storage, read_write> tile_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> face_ids: array<u32>;
@group(0) @binding(5) var<storage, read> tetra_uvs: array<mat4x2<f32>>;
@group(0) @binding(6) var<uniform> num_tetra: u32;
@group(0) @binding(7) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(8) var<uniform> tile_size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > num_tetra * 4) {
        return;
    }
    let indices = array<vec4<u32>, 3>(
        vec4<u32>(0, 0, 0, 1),
        vec4<u32>(1, 3, 2, 3),
        vec4<u32>(2, 1, 3, 2),
    );
    let tetra_id = global_id.x / u32(4);
    let face_id = global_id.x % 4;
    let rect = face_rects[global_id.x];
    let num_tiles = vec2<f32>(ceil(f32(canvas_size.x) / f32(tile_size)), ceil(f32(canvas_size.y) / f32(tile_size)));
    var offs = tile_offsets[global_id.x];
    for (var y = rect.y; y < rect.w; y++) {
        for (var x = rect.x; x < rect.z; x++) {
            let tile_id : u32 = y * u32(num_tiles.x) + x;
            let depth = trilinear_interpolate(
                vec2<f32>((f32(x) + 0.5) / num_tiles.x, (f32(y) + 0.5) / num_tiles.y),
                tetra_uvs[tetra_id][indices[0][face_id]],
                tetra_uvs[tetra_id][indices[1][face_id]],
                tetra_uvs[tetra_id][indices[2][face_id]],
                face_depths[global_id.x][0],
                face_depths[global_id.x][1],
                face_depths[global_id.x][2],
            );
            // TODO: Fix when this overflows for large number of tiles
            // TODO: assumes depths are 0-5
            tile_ids[offs] = tile_id * 1000 + u32(min(200 * depth, 999));
            face_ids[offs] = global_id.x;
            offs++;
        }
    }
}

fn trilinear_interpolate(
    p: vec2<f32>,
    a: vec2<f32>,
    b: vec2<f32>,
    c: vec2<f32>,
    f1: f32,
    f2: f32,
    f3: f32
) -> f32 {
    let v0 = b - a;
    let v1 = c - a;
    let v2 = p - a;
    let d00 = dot(v0, v0);
    let d01 = dot(v0, v1);
    let d11 = dot(v1, v1);
    let d20 = dot(v2, v0);
    let d21 = dot(v2, v1);
    let denom = d00 * d11 - d01 * d01;
    let v = (d11 * d20 - d01 * d21) / denom;
    let w = (d00 * d21 - d01 * d20) / denom;
    let u = 1.0 - v - w;
    let z = (1.0 / f1) * u + (1.0 / f2) * v + (1.0 / f3) * w;
    // return (1.0 / z);
    return (f1 + f2 + f3) / 3;
}