@group(0) @binding(0) var<storage, read_write> vals: array<u32>;
@group(0) @binding(1) var<storage, read_write> block_sums: array<u32>;

var<workgroup> chunk : array<u32, 512>;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, @builtin(local_invocation_id) local_id : vec3<u32>, @builtin(workgroup_id) w_id: vec3<u32>) {
    const BLOCK_SIZE : u32 = 512;
    chunk[2 * local_id.x] = vals[2 * global_id.x];
    chunk[2 * local_id.x + 1] = vals[2 * global_id.x + 1];

    var offs : u32 = 1;
    // Reduce step up tree
    for (var d = BLOCK_SIZE >> 1; d > 0; d = d >> 1) {
        workgroupBarrier();
        if (local_id.x < d) {
            let a = offs * (2 * local_id.x + 1) - 1;
            let b = offs * (2 * local_id.x + 2) - 1;
            chunk[b] += chunk[a];
        }
        offs = offs << 1;
    }

    if (local_id.x == 0) {
        block_sums[w_id.x] = chunk[BLOCK_SIZE - 1];
        chunk[BLOCK_SIZE - 1] = 0;
    }

    // Sweep down the tree to finish the scan
    for (var d : u32 = 1; d < BLOCK_SIZE; d = d << 1) {
        offs = offs >> 1;
        workgroupBarrier();
        if (local_id.x < d) {
            let a = offs * (2 * local_id.x + 1) - 1;
            let b = offs * (2 * local_id.x + 2) - 1;
            let tmp = chunk[a];
            chunk[a] = chunk[b];
            chunk[b] += tmp;
        }
    }

    workgroupBarrier();
    vals[2 * global_id.x] = chunk[2 * local_id.x];
    vals[2 * global_id.x + 1] = chunk[2 * local_id.x + 1];
}

