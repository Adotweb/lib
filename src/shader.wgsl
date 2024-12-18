// Vertex shader

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) tex_coords: vec2<f32>,
    @location(2) color : vec3<f32>
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) tex_coords: vec2<f32>,
    @location(1) color : vec3<f32>
}

// Bind Group 1
struct Uniforms {
    time: f32
};

@group(1) @binding(0)
var<uniform> myUniforms: Uniforms;

@vertex
fn vs_main(
    model: VertexInput,
) -> VertexOutput {
    var out: VertexOutput;
    out.tex_coords = model.tex_coords;
    out.clip_position = vec4<f32>(model.position.x, model.position.yz, 1.0);
    out.color = model.color;
    return out;
}
// Fragment shader

@group(0) @binding(0)
var t_diffuse: texture_2d<f32>;
@group(0) @binding(1)
var s_diffuse: sampler;

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    let tex_col = textureSample(t_diffuse, s_diffuse, in.tex_coords);

    let fin = tex_col * vec4<f32>(in.color, 1.0) * sin(myUniforms.time);

    return fin;
}
