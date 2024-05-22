struct GaussianData {
    uv: vec2<f32>,
    conic: vec3<f32>,
    depth: u32,
    color: vec3<f32>,
    opacity: f32,
    rect: vec4<u32>,
}

@group(0) @binding(0) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> gaussian_data: array<GaussianData>;
@group(0) @binding(2) var<storage, read_write> tile_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> gaussian_ids: array<u32>;
@group(0) @binding(4) var<uniform> n_unpadded: u32;
@group(0) @binding(5) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(6) var<uniform> tile_size: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > n_unpadded) {
        return;
    }
    let gaussian = gaussian_data[global_id.x];
    let num_tiles = vec2<f32>(ceil(f32(canvas_size.x) / f32(tile_size)), ceil(f32(canvas_size.y) / f32(tile_size)));
    var offs = tile_offsets[global_id.x];
    for (var y = gaussian.rect.y; y < gaussian.rect.w; y++) {
        for (var x = gaussian.rect.x; x < gaussian.rect.z; x++) {
            let tile_id = y * u32(num_tiles.x) + x;
            // TODO: Fix when this overflows for large number of tiles
            tile_ids[offs] = tile_id * 1000 + gaussian.depth;
            gaussian_ids[offs] = global_id.x;
            offs++;
        }
    }
}