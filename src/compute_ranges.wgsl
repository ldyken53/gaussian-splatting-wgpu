@group(0) @binding(0) var<storage, read_write> tile_ids: array<u32>;
@group(0) @binding(1) var<storage, read_write> ranges: array<u32>;
@group(0) @binding(2) var<uniform> num_tiles: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    var start_index : u32;
    if (global_id.x == 0) {
        start_index = 0;
    } else {
        start_index = tile_ids[global_id.x * 64 - 1] / 1000u;
    }
    if (tile_ids[global_id.x * 64] == 4294967294u) {
        return;
    }
    for (var idx = global_id.x * 64; idx < global_id.x * 64 + 64; idx++) {
        if (tile_ids[idx] / 1000 > start_index) {
            for (var i : u32 = start_index; i < tile_ids[idx] / 1000u; i++) {
                ranges[i] = idx;
            }
        }
        if (tile_ids[idx + 1] == 4294967294u) {
            for (var i = tile_ids[idx] / 1000u; i < num_tiles; i++) {
                ranges[i] = idx + 1;
            }
            return;
        }
        start_index = tile_ids[idx] / 1000u;
    }
}