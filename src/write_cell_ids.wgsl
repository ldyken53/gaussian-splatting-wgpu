struct AABBs {
    start_cell: vec3<u32>,
    end_cell: vec3<u32>,
    value: f32
};
struct Uniforms {
    volume_mins: vec3<f32>,
    cell_size: f32,
    volume_maxes: vec3<f32>,
};

@group(0) @binding(0) var<storage, read> cell_offsets: array<u32>;
@group(0) @binding(1) var<storage, read> gaussian_data: array<AABBs>;
@group(0) @binding(2) var<storage, read_write> cell_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> gaussian_ids: array<f32>;
@group(0) @binding(4) var<uniform> n_unpadded: u32;
@group(0) @binding(5) var<uniform> uniforms: Uniforms;


@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if (global_id.x > n_unpadded) {
        return;
    }
    let gaussian = gaussian_data[global_id.x];
    let num_cells = vec3<u32>(ceil((uniforms.volume_maxes - uniforms.volume_mins) / uniforms.cell_size));
    var offs = cell_offsets[global_id.x];
    for (var z = gaussian.start_cell.z; z < gaussian.end_cell.z; z += 1) {
        for (var y = gaussian.start_cell.y; y < gaussian.end_cell.y; y += 1) {
            for (var x = gaussian.start_cell.x; x < gaussian.end_cell.x; x += 1) {
                cell_ids[offs] = z * num_cells.x * num_cells.y + y * num_cells.x + x;
                gaussian_ids[offs] = gaussian.value;
                offs++;
            }
        }
    }
}