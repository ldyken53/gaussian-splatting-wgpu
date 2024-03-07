@group(0) @binding(0) var<uniform> size : u32;
@group(1) @binding(0) var<storage, read_write> keys : array<u32>;
@group(1) @binding(1) var<storage, read_write> values : array<u32>;
@group(2) @binding(0) var<uniform> work_group_offset : u32;

var<workgroup> key_buf : array<u32, 256>;
var<workgroup> sorted_key_buf : array<u32, 256>;
var<workgroup> scratch : array<u32, 256>;
var<workgroup> total_false : u32;

var<workgroup> val_buf : array<u32, 256>;
var<workgroup> sorted_val_buf : array<u32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>) {
    const UINT_MAX : u32 = 4294967295u;
    // Also use the radix step to pad arrays out with UINT_MAX
    let item_idx : u32 = work_group_offset * 256 + global_id.x;
    if (item_idx < size) {
        key_buf[local_id.x] = keys[item_idx];
        val_buf[local_id.x] = values[item_idx];
    } else {
        // Pad any missing data with uint max, which will be sorted out to the end
        key_buf[local_id.x] = UINT_MAX;
        val_buf[local_id.x] = UINT_MAX;
    }

    // Sort each bit, from LSB to MSB
    for (var i : u32 = 0; i < 32; i++) {
        workgroupBarrier();
        let mask : u32 = 1u << i;
        scratch[local_id.x] = select(1u, 0u, (key_buf[local_id.x] & mask) != 0);
        
        // A bit annoying to copy this code around, but we can't have unsized array
        // parameters to functions in GLSL
        var offs : u32 = 1;
        // Reduce step up tree
        for (var d : u32 = 256 >> 1; d > 0; d = d >> 1) {
            workgroupBarrier();
            if (local_id.x < d) {
                let a : u32 = offs * (2 * local_id.x + 1) - 1;
                let b : u32 = offs * (2 * local_id.x + 2) - 1;
                scratch[b] += scratch[a];
            }
            offs = offs << 1;
        }

        if (local_id.x == 0) {
            total_false = scratch[256 - 1];
            scratch[256 - 1] = 0;
        }

        // Sweep down the tree to finish the scan
        for (var d : u32 = 1; d < 256; d = d << 1) {
            offs = offs >> 1;
            workgroupBarrier();
            if (local_id.x < d) {
                var a : u32 = offs * (2 * local_id.x + 1) - 1;
                var b : u32 = offs * (2 * local_id.x + 2) - 1;
                let tmp : u32 = scratch[a];
                scratch[a] = scratch[b];
                scratch[b] += tmp;
            }
        }
        workgroupBarrier();

        // Now scatter the elements to their destinations
        let f : u32 = scratch[local_id.x];
        let t : u32 = local_id.x - f + total_false;
        if ((key_buf[local_id.x] & mask) != 0) {
            sorted_key_buf[t] = key_buf[local_id.x];
            sorted_val_buf[t] = val_buf[local_id.x];
        } else {
            sorted_key_buf[f] = key_buf[local_id.x];
            sorted_val_buf[f] = val_buf[local_id.x];
        }
        workgroupBarrier();

        // Copy the sorted set to the buf for the next pass
        key_buf[local_id.x] = sorted_key_buf[local_id.x];
        val_buf[local_id.x] = sorted_val_buf[local_id.x];
    }
    workgroupBarrier();
    
    // Write back the sorted buffer
    keys[item_idx] = key_buf[local_id.x];
    values[item_idx] = val_buf[local_id.x];
}