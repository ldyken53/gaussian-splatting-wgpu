struct Uniforms {
    volume_mins: vec3<f32>,
    cell_size: f32,
    volume_maxes: vec3<f32>,
};
@group(0) @binding(0) var<storage, read_write> cell_data: array<f32>;
@group(0) @binding(1) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> gaussian_values: array<f32>;
@group(0) @binding(3) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_cells = vec3<u32>(ceil((uniforms.volume_maxes - uniforms.volume_mins) / uniforms.cell_size));
    if (global_id.x > num_cells.x * num_cells.y * num_cells.z) {
        return;
    }
    let cell_id = global_id.x;
    let gaussians_per_cell = f32(ranges[cell_id].y - ranges[cell_id].x);
    var accumulated_value = 0.0;  
    for (var i = ranges[cell_id].x; i < ranges[cell_id].y; i++) {
        let value = gaussian_values[i];
        accumulated_value += (value / gaussians_per_cell);
    }
    cell_data[cell_id] = accumulated_value;
}