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

// MLP stage (trained kernels): a per-cell two-layer MLP
// (hidden ReLU layer, then an output layer) applied to the conv results
// before the activation, giving the update rule a genuine nonlinearity.
@useMlpFlag
const MLP_HIDDEN: u32 = @mlpHiddenu;
// MLP input width: the CHANNEL_COUNT conv results, doubled when the kernel
// was trained with the cell's own raw state as extra MLP inputs (stateInput)
const MLP_IN: u32 = @mlpInu;
// Offsets into the flat mlpWeights buffer:
// [w1: MLP_HIDDEN x MLP_IN][b1: MLP_HIDDEN][w2: CHANNEL_COUNT x MLP_HIDDEN][b2: CHANNEL_COUNT]
const MLP_B1_OFFSET: u32 = MLP_HIDDEN * MLP_IN;
const MLP_W2_OFFSET: u32 = MLP_B1_OFFSET + MLP_HIDDEN;
const MLP_B2_OFFSET: u32 = MLP_W2_OFFSET + CHANNEL_COUNT * MLP_HIDDEN;

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

// Per-cell MLP weights, used when USE_MLP (see the offset constants above
// for the layout); a minimal placeholder buffer is bound otherwise
@group(0) @binding(6)
var<storage, read> mlpWeights: array<f32>;

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

  if (USE_MLP) {
    // Per-cell MLP between the conv results and the activation:
    // hidden layer relu(w1 . sums + b1) ...
    var hiddenAct: array<f32, MLP_HIDDEN>;
    for (var j: u32 = 0u; j < MLP_HIDDEN; j++) {
      var acc: f32 = mlpWeights[MLP_B1_OFFSET + j];
      for (var ch: u32 = 0u; ch < CHANNEL_COUNT; ch++) {
        acc += mlpWeights[j * MLP_IN + ch] * sums[ch];
      }
      // stateInput kernels: the cell's own raw state follows the conv
      // results as a second block of w1 columns (const condition, so the
      // branch compiles out for conv-only-input kernels)
      if (MLP_IN > CHANNEL_COUNT) {
        for (var ch: u32 = 0u; ch < CHANNEL_COUNT; ch++) {
          acc += mlpWeights[j * MLP_IN + CHANNEL_COUNT + ch] * activationContext.cellState[ch];
        }
      }
      hiddenAct[j] = max(acc, 0.0);
    }

    // ... then the output layer becomes the per-channel update signal,
    // handed to the exported activation exactly like the raw convX
    var mlpOut: array<f32, CHANNEL_COUNT>;
    for (var outCh: u32 = 0u; outCh < CHANNEL_COUNT; outCh++) {
      var acc: f32 = mlpWeights[MLP_B2_OFFSET + outCh];
      for (var j: u32 = 0u; j < MLP_HIDDEN; j++) {
        acc += mlpWeights[MLP_W2_OFFSET + outCh * MLP_HIDDEN + j] * hiddenAct[j];
      }
      mlpOut[outCh] = acc;
      totalWeights[outCh] = 1.0;
    }
    sums = mlpOut;
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
