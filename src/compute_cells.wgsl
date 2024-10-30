struct Gaussian {
    conic: array<f32, 6>,
    start_cell: vec3<u32>,
    det: f32,
    end_cell: vec3<u32>,
    value: f32,
    mean: vec3<f32>
};
struct Uniforms {
    volume_mins: vec3<f32>,
    cell_size: f32,
    volume_maxes: vec3<f32>,
};
@group(0) @binding(0) var<storage, read_write> cell_data: array<f32>;
@group(0) @binding(1) var<storage, read> ranges: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> gaussian_ids: array<u32>;
@group(0) @binding(3) var<storage, read> gaussian_data: array<Gaussian>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let num_cells = vec3<u32>(ceil((uniforms.volume_maxes - uniforms.volume_mins) / uniforms.cell_size));
    if (global_id.x >= num_cells.x * num_cells.y * num_cells.z) {
        return;
    }
    let cell_id = global_id.x;
    let cell = vec3<u32>(
        (cell_id % (num_cells.x * num_cells.y)) % num_cells.x,
        (cell_id % (num_cells.x * num_cells.y)) / num_cells.x,
        cell_id / (num_cells.x * num_cells.y)
    );
    let cell_pos = (vec3<f32>(cell) + vec3<f32>(0.5)) * uniforms.cell_size + uniforms.volume_mins;
    let gaussians_per_cell = f32(ranges[cell_id].y - ranges[cell_id].x);
    let scale_modifier = 1.0;
    var accumulated_value = 0.0;  
    var accumulated_weight = 0.0;
    for (var i = ranges[cell_id].x; i < ranges[cell_id].y; i++) {
        let gaussian = gaussian_data[gaussian_ids[i]];
        let delta = cell_pos - gaussian.mean;
        let conic = gaussian.conic;
        let quad_form = (
            delta.x * (conic[0] * delta.x + conic[1] * delta.y + conic[2] * delta.z) +
            delta.y * (conic[1] * delta.x + conic[3] * delta.y + conic[4] * delta.z) +
            delta.z * (conic[2] * delta.x + conic[4] * delta.y + conic[5] * delta.z)
        );
        // Normalize makes Gaussians have equal weight integral of 1
        // over the entire volume, regardless of scale
        // i.e. smaller Gaussians have higher peak weight
        // let normalize_factor = 1.0 / (pow(2.0 * 3.14159, 1.5) * sqrt(gaussian.det));
        let normalize_factor = 1.0;
        let weight = normalize_factor * exp(-0.5 * quad_form * scale_modifier);
        // if (exp(-0.5 * quad_form * scale_modifier) > 1) {
        //     // Break on purpose, numerical issue !!!
        //     cell_data[cell_id] = -1;
        //     return;
        // }
        accumulated_value += gaussian.value * weight;
        accumulated_weight += weight;
    }
    // This both gives a dropoff where we have to have a certain weight to set a value
    // and prevents numerical issues of dividing by something close to 0
    if (accumulated_weight > 1e-5) {
        cell_data[cell_id] = accumulated_value / accumulated_weight;
    } else {
        cell_data[cell_id] = 0.0;
    }
}