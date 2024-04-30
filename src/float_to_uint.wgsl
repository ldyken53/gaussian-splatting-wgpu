@group(0) @binding(0) var<storage, read_write> arr: array<u32>;

fn floatFlip(f : u32) -> u32 {
    let mask : u32 = (u32(-(i32(f >> 31))) | 0x80000000u);
    return f ^ mask;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    arr[global_id.x] = floatFlip(arr[global_id.x]);
}
