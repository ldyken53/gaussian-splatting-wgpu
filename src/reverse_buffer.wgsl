@group(0) @binding(0) var<uniform> size : u32;
@group(1) @binding(0) var<storage, read_write> values : array<u32>;
@group(2) @binding(0) var<uniform> work_group_offset : u32;

fn next_pow2(x : u32) -> u32 {
    var y : u32 = x - 1;
    y |= y >> 1;
    y |= y >> 2;
    y |= y >> 4;
    y |= y >> 8;
    y |= y >> 16;
    return y + 1;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
    // Each thread swaps a pair of elements in place
    let aligned_size : u32 = next_pow2(u32(ceil(f32(size) / 256))) * 256;
    // For buffers < SORT_CHUNK_SIZE don't swap elements out of bounds
    if (aligned_size < 256 && global_id.x > 256 / 2) {
        return;
    }
    let idx : u32 = work_group_offset * 256 + global_id.x;
    let i : u32 = idx;
    let j : u32 = aligned_size - idx - 1;
    let tmp : u32 = values[i];
    values[i] = values[j];
    values[j] = tmp;
}