// Constants
const WIDTH: u32 = @sizeWidthu;
const HEIGHT: u32 = @sizeHeightu;
const KERNEL_SIZE: i32 = 5;
const HALF_KERNEL: i32 = KERNEL_SIZE / 2;
const CHANNEL_COUNT: u32 = 3u;
const KERNEL_AREA: u32 = u32(KERNEL_SIZE * KERNEL_SIZE);

@computeKernelFlag

// Bindings
@group(0) @binding(0)
var src: texture_2d<f32>;

@group(0) @binding(1)
var dst: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<storage, read> weightBuffer: array<f32>;

@group(0) @binding(3)
var<uniform> timestepData: vec4<f32>; // [timestep, clickX, clickY, unused]

// Coordinate wrapping
fn wrapCoord(x: i32, y: i32) -> vec2<i32> {
  let ix = (x + i32(WIDTH)) % i32(WIDTH);
  let iy = (y + i32(HEIGHT)) % i32(HEIGHT);
  return vec2<i32>(ix, iy);
}

// Additional values that can be accessed by the activation function
struct ActivationContext {
    gid: vec3<u32>,
    weightSum: f32,
    cellState: vec3<f32>,
    channel: u32,
    timestep: f32,
    clickX: f32,
    clickY: f32
}

var<private> activationContext: ActivationContext;

// Activation function placholder
@activationFunction

fn activationClamped(x: f32, weightSum: f32) -> f32 {
  activationContext.weightSum = weightSum;

  // Flag to control normalization 
  @normalizeFlag

  return clamp(activation(norm), 0.0, 1.0);
}

// Get weight from flat array
fn getWeight(outCh: u32, inCh: u32, kernelIndex: u32) -> f32 {
  return weightBuffer[outCh * 75u + inCh * 25u + kernelIndex];
}

// Main compute shader
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  activationContext.gid = gid;
  activationContext.cellState = textureLoad(src, vec2<u32>(gid.x, gid.y), 0).rgb;
  
  let x: i32 = i32(gid.x);
  let y: i32 = i32(gid.y);

  var sumR: f32 = 0.0;
  var sumG: f32 = 0.0;
  var sumB: f32 = 0.0;

  var totalWeightR: f32 = 0.0;
  var totalWeightG: f32 = 0.0;
  var totalWeightB: f32 = 0.0;

  if (COMPUTE_KERNEL) {
    for (var ky: i32 = -HALF_KERNEL; ky <= HALF_KERNEL; ky++) {
      for (var kx: i32 = -HALF_KERNEL; kx <= HALF_KERNEL; kx++) {
        let coord = wrapCoord(x + kx, y + ky);
        let pixel = textureLoad(src, vec2<u32>(coord), 0).rgb;

        let wx = kx + HALF_KERNEL;
        let wy = ky + HALF_KERNEL;
        let kernelIndex = u32(wy * KERNEL_SIZE + wx);

        // Weights for each output channel
        let wRR = getWeight(0u, 0u, kernelIndex);
        let wRG = getWeight(0u, 1u, kernelIndex);
        let wRB = getWeight(0u, 2u, kernelIndex);

        let wGR = getWeight(1u, 0u, kernelIndex);
        let wGG = getWeight(1u, 1u, kernelIndex);
        let wGB = getWeight(1u, 2u, kernelIndex);

        let wBR = getWeight(2u, 0u, kernelIndex);
        let wBG = getWeight(2u, 1u, kernelIndex);
        let wBB = getWeight(2u, 2u, kernelIndex);

        sumR += pixel.r * wRR + pixel.g * wRG + pixel.b * wRB;
        sumG += pixel.r * wGR + pixel.g * wGG + pixel.b * wGB;
        sumB += pixel.r * wBR + pixel.g * wBG + pixel.b * wBB;

        totalWeightR += wRR + wRG + wRB;
        totalWeightG += wGR + wGG + wGB;
        totalWeightB += wBR + wBG + wBB;
      }
    }
  }

  activationContext.timestep = timestepData.x;
  activationContext.clickX = timestepData.y;
  activationContext.clickY = timestepData.z;

  activationContext.channel = 0;
  let outR = activationClamped(sumR, totalWeightR);
  activationContext.channel = 1;
  let outG = activationClamped(sumG, totalWeightG);
  activationContext.channel = 2;
  let outB = activationClamped(sumB, totalWeightB);

  textureStore(dst, vec2<u32>(gid.xy), vec4<f32>(outR, outG, outB, 1.0));
}
