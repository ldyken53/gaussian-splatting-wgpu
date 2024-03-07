@group(0) @binding(0) var<uniform> size : u32;
@group(1) @binding(0) var<storage, read_write> input_keys : array<u32>;
@group(1) @binding(1) var<storage, read_write> input_values : array<u32>;
@group(1) @binding(2) var<storage, read_write> output_keys : array<u32>;
@group(1) @binding(3) var<storage, read_write> output_values : array<u32>;
@group(2) @binding(0) var<uniform> work_groups_x : u32;
@group(3) @binding(0) var<uniform> work_group_offset : u32;

fn upper_bound(start : u32, count : u32, element : u32) -> u32 {
    var c : u32 = count;
    var s : u32 = start;
    while (c > 0) {
        var i : u32 = s + c / 2;
        if (element >= input_keys[i]) {
            s = i + 1;
            c -= c / 2 + 1;
        } else {
            c = c / 2;
        }
    }
    return s;
}

fn lower_bound(start : u32, count : u32, element : u32) -> u32 {
    var c : u32 = count;
    var s : u32 = start;
    while (c > 0) {
        var i : u32 = s + c / 2;
        if (input_keys[i] < element) {
            s = i + 1;
            c -= c / 2 + 1;
        } else {
            c = c / 2;
        }
    }
    return s;
}

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
fn main(@builtin(workgroup_id) wg_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) {
    // Compute the merge chunk size, which is based on the number of work groups and input data size
    let aligned_size : u32 = next_pow2(u32(ceil(f32(size) / 256))) * 256;
    let merge_output_size : u32 = aligned_size / work_groups_x;
    let merge_chunk_size : u32 = merge_output_size / 2;

    // Load the first set of elements to merge
    let offs : u32 = (work_group_offset + wg_id.x) * merge_output_size;

    // Each work group merges two chunks, each thread is responsible for
    // two elements in the chunks, which it merges into the sorted output
    // Loop through and merge each SORT_CHUNK_SIZE group of elements from merge_chunk_size
    for (var i : u32 = 0; i < merge_chunk_size / 256; i++) {
        let a_in : u32 = offs + i * 256 + local_id.x;
        let b_in : u32 = offs + merge_chunk_size + i * 256 + local_id.x;
        let base_idx : u32 = local_id.x + i * 256;
        // Could be done better, but short on time 
        // Upper bound in b
        let a_loc : u32 = base_idx
            + upper_bound(offs + merge_chunk_size, merge_chunk_size, input_keys[a_in])
            - merge_chunk_size;
        // Lower bound in a
        let b_loc : u32 = base_idx + lower_bound(offs, merge_chunk_size, input_keys[b_in]);

        output_keys[a_loc] = input_keys[a_in];
        output_values[a_loc] = input_values[a_in];

        output_keys[b_loc] = input_keys[b_in];
        output_values[b_loc] = input_values[b_in];
    }
}



