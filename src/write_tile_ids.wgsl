@group(0) @binding(0) var<storage, read> tile_offsets: array<u32>;
@group(0) @binding(1) var<storage, read_write> tetra_rects: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> tetra_depths: array<f32>;
@group(0) @binding(3) var<storage, read_write> tile_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> tetra_ids: array<u32>;
@group(0) @binding(5) var<uniform> num_tetra: u32;
@group(0) @binding(6) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(7) var<uniform> tile_size: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > num_tetra) {
        return;
    }
    let rect = tetra_rects[global_id.x];
    let depth = tetra_depths[global_id.x];
    let num_tiles = vec2<f32>(ceil(f32(canvas_size.x) / f32(tile_size)), ceil(f32(canvas_size.y) / f32(tile_size)));
    var offs = tile_offsets[global_id.x];
    for (var y = rect.y; y < rect.w; y++) {
        for (var x = rect.x; x < rect.z; x++) {
            let tile_id : u32 = y * u32(num_tiles.x) + x;
            // TODO: Fix when this overflows for large number of tiles
            // TODO: assumes depths are 0-99.9
            tile_ids[offs] = tile_id * 1000 + u32(min(10 * depth, 999));
            tetra_ids[offs] = global_id.x;
            offs++;
        }
    }
}