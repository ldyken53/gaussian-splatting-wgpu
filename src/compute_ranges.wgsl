@group(0) @binding(0) var<storage, read_write> tiles: array<u32>;
@group(0) @binding(1) var<storage, read_write> ranges: array<u32>;
@group(0) @binding(2) var<uniform> num_tiles: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (tiles[global_id.x] == 4294967294u) {
        return;
    }
    var start_index : u32;
    if (global_id.x == 0) {
        start_index = 0;
    } else {
        start_index = tiles[global_id.x - 1];
    }
    if (tiles[global_id.x] > start_index) {
        for (var i : u32 = start_index; i < tiles[global_id.x]; i++) {
            ranges[i] = global_id.x;
        }
    }
    if (tiles[global_id.x + 1] == 4294967294u) {
        for (var i = tiles[global_id.x]; i < num_tiles; i++) {
            ranges[i] = global_id.x + 1;
        }
    }
}