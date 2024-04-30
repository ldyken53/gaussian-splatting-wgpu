@group(0) @binding(0) var<storage, read_write> vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> depths: array<f32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<uniform> proj_matrix: mat4x4<f32>;
@group(0) @binding(4) var<uniform> n_unpadded: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    indices[global_id.x] = global_id.x;
    if (global_id.x >= n_unpadded) {
        depths[global_id.x] = 1e20f; // pad with +inf
    } else {
        let pos = vertices[global_id.x];
        let proj_pos = proj_matrix * vec4<f32>(pos, 1.0);
        depths[global_id.x] = proj_pos.z;
    }
}