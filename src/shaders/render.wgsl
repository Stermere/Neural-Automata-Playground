// Vertex stage
@vertex
fn vs(@builtin(vertex_index) i : u32)
    -> @builtin(position) vec4<f32> {
    var pos = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0,  1.0)
    );
    return vec4<f32>(pos[i], 0.0, 1.0);
}

// The texture we're sampling
@group(0) @binding(0) var img : texture_2d<f32>;

// A sampler for that texture
@group(0) @binding(1) var linearSampler : sampler;

@fragment
fn fs(@builtin(position) coord : vec4<f32>)
    -> @location(0) vec4<f32> {
    let uv = coord.xy / vec2<f32>(1024.0, 1024.0);
    // use our sampler here
    return textureSample(img, linearSampler, uv);
}