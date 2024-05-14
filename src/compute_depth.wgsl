struct Uniforms {
    view_matrix: mat4x4<f32>,
    proj_matrix: mat4x4<f32>,
    camera_position: vec3<f32>,
    tan_fovx: f32,
    tan_fovy: f32,
    focal_x: f32,
    focal_y: f32,
    scale_modifier: f32,
};

@group(0) @binding(0) var<storage, read_write> vertices: array<vec3<f32>>;
@group(0) @binding(1) var<storage, read_write> depths: array<f32>;
@group(0) @binding(2) var<storage, read_write> indices: array<u32>;
@group(0) @binding(3) var<storage, read_write> tiles: array<u32>;
@group(0) @binding(4) var<uniform> uniforms: Uniforms;
@group(0) @binding(5) var<uniform> n_unpadded: u32;
@group(0) @binding(6) var<uniform> canvas_size: vec2<u32>;
@group(0) @binding(7) var<uniform> tile_size: u32;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    indices[global_id.x] = global_id.x;
    if (global_id.x >= n_unpadded) {
        depths[global_id.x] = 1e20f; // pad with +inf
        tiles[global_id.x] = 4294967294u;
    } else {
        let pos = vertices[global_id.x];
        let proj_pos = uniforms.view_matrix * vec4<f32>(pos, 1.0);
        depths[global_id.x] = proj_pos.z;
        if (proj_pos.x < 0 || proj_pos.x >= 1 || proj_pos.y < 0 || proj_pos.y >= 1) {
            tiles[global_id.x] = 4294967294u;
        } else {
            let t = tile_size;
            let num_tiles = vec2<f32>(ceil(f32(canvas_size.x) / f32(tile_size)), ceil(f32(canvas_size.y) / f32(tile_size)));
            tiles[global_id.x] = u32(proj_pos.x * num_tiles.x) + u32(floor(proj_pos.y * num_tiles.y) * num_tiles.x); 
        }
    }
}