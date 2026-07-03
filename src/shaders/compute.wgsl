// Constants
const WIDTH: u32 = @sizeWidthu;
const HEIGHT: u32 = @sizeHeightu;
const KERNEL_SIZE: i32 = 5;
const HALF_KERNEL: i32 = KERNEL_SIZE / 2;
const CHANNEL_COUNT: u32 = @channelCountu; // visible + hidden
const VISIBLE_COUNT: u32 = 3u;
const HIDDEN_COUNT: u32 = @hiddenCountu;
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

// Hidden channel state (HIDDEN_COUNT floats per cell), ping-ponged like src/dst
@group(0) @binding(4)
var<storage, read> hiddenSrc: array<f32>;

@group(0) @binding(5)
var<storage, read_write> hiddenDst: array<f32>;

// Coordinate wrapping
fn wrapCoord(x: i32, y: i32) -> vec2<i32> {
  let ix = (x + i32(WIDTH)) % i32(WIDTH);
  let iy = (y + i32(HEIGHT)) % i32(HEIGHT);
  return vec2<i32>(ix, iy);
}

fn hiddenIndex(coord: vec2<i32>, ch: u32) -> u32 {
  return (u32(coord.y) * WIDTH + u32(coord.x)) * HIDDEN_COUNT + ch;
}

// Load the full cell state (visible + hidden channels) at a wrapped coordinate
fn loadState(coord: vec2<i32>) -> array<f32, CHANNEL_COUNT> {
  var state: array<f32, CHANNEL_COUNT>;
  let pixel = textureLoad(src, vec2<u32>(coord), 0);
  state[0] = pixel.r;
  state[1] = pixel.g;
  state[2] = pixel.b;
  for (var h: u32 = 0u; h < HIDDEN_COUNT; h++) {
    state[VISIBLE_COUNT + h] = hiddenSrc[hiddenIndex(coord, h)];
  }
  return state;
}

// Additional values that can be accessed by the activation function
struct ActivationContext {
    gid: vec3<u32>,
    weightSum: f32,
    cellState: array<f32, CHANNEL_COUNT>,
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

// Get weight from flat array laid out as [outChannel][inChannel][kernelIndex]
fn getWeight(outCh: u32, inCh: u32, kernelIndex: u32) -> f32 {
  return weightBuffer[(outCh * CHANNEL_COUNT + inCh) * KERNEL_AREA + kernelIndex];
}

// Main compute shader
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= WIDTH || gid.y >= HEIGHT) {
    return;
  }

  activationContext.gid = gid;

  let x: i32 = i32(gid.x);
  let y: i32 = i32(gid.y);
  let selfCoord = vec2<i32>(x, y);
  activationContext.cellState = loadState(selfCoord);

  var sums: array<f32, CHANNEL_COUNT>;
  var totalWeights: array<f32, CHANNEL_COUNT>;

  if (COMPUTE_KERNEL) {
    for (var ky: i32 = -HALF_KERNEL; ky <= HALF_KERNEL; ky++) {
      for (var kx: i32 = -HALF_KERNEL; kx <= HALF_KERNEL; kx++) {
        let coord = wrapCoord(x + kx, y + ky);
        var state = loadState(coord);

        let wx = kx + HALF_KERNEL;
        let wy = ky + HALF_KERNEL;
        let kernelIndex = u32(wy * KERNEL_SIZE + wx);

        for (var outCh: u32 = 0u; outCh < CHANNEL_COUNT; outCh++) {
          for (var inCh: u32 = 0u; inCh < CHANNEL_COUNT; inCh++) {
            let w = getWeight(outCh, inCh, kernelIndex);
            sums[outCh] += state[inCh] * w;
            totalWeights[outCh] += w;
          }
        }
      }
    }
  }

  activationContext.timestep = timestepData.x;
  activationContext.clickX = timestepData.y;
  activationContext.clickY = timestepData.z;

  var outState: array<f32, CHANNEL_COUNT>;
  for (var ch: u32 = 0u; ch < CHANNEL_COUNT; ch++) {
    activationContext.channel = ch;
    outState[ch] = activationClamped(sums[ch], totalWeights[ch]);
  }

  textureStore(dst, vec2<u32>(gid.xy), vec4<f32>(outState[0], outState[1], outState[2], 1.0));
  for (var h: u32 = 0u; h < HIDDEN_COUNT; h++) {
    hiddenDst[hiddenIndex(selfCoord, h)] = outState[VISIBLE_COUNT + h];
  }
}
