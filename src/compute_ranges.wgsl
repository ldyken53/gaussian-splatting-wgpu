@group(0) @binding(0) var<storage, read_write> tile_ids: array<u32>;
@group(0) @binding(1) var<storage, read_write> ranges: array<vec2<u32>>;
@group(0) @binding(2) var<uniform> num_intersections: u32;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if (index >= num_intersections) {
        return;
    }
    let current_tile = tile_ids[index];
    if (index == 0) {
        ranges[current_tile].x = 0;
    } else {
        let previous_tile = tile_ids[index - 1];
        if (current_tile > previous_tile) {
            ranges[previous_tile].y = index;
            ranges[current_tile].x = index;
        }
    }
    if (index == num_intersections - 1) {
        ranges[current_tile].y = num_intersections;
    }
}