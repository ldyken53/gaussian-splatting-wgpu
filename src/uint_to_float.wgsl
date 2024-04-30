@group(0) @binding(0) var<storage, read_write> arr: array<u32>;

fn inverseFloatFlip(f : u32) -> u32 {
    let mask : u32 = ((f >> 31) - 1u) | 0x80000000u;
    return f ^ mask;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    arr[global_id.x] = inverseFloatFlip(arr[global_id.x]);
}
