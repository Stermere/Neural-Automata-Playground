export const BASE_ACTIVATIONS: Record<string, string> = {
  "Brightness Switching": `const BRIGHT_THRESHOLD: f32 = 0.5; @variable 0.0 1.0
const DARK_GROWTH: f32 = 0.04; @variable -0.5 0.5
const BRIGHT_DECAY: f32 = -0.05; @variable -0.5 0.5

fn activation(x: f32) -> f32 {
  let brightness = (activationContext.cellState.r + activationContext.cellState.g + activationContext.cellState.b) / 3.0;

  if (brightness < BRIGHT_THRESHOLD) {
      return clamp(activationContext.cellState[activationContext.channel] + DARK_GROWTH * tanh(x), 0.0, 1.0);
  } else {
      return clamp(activationContext.cellState[activationContext.channel] + BRIGHT_DECAY * tanh(x), 0.0, 1.0);
  }
}`,
  
  "Radial Influence": `const CENTER_X: f32 = f32(WIDTH) / 2.0;
const CENTER_Y: f32 = f32(HEIGHT) / 2.0;
const RADIUS_SCALE: f32 = 0.04; @variable 0.0 1.0
const BASE_RATE: f32 = 0.04; @variable 0.0 2.0
const TIME_SCALE: f32 = 10; @variable 10 500

fn activation(x: f32) -> f32 {
  let dx = f32(activationContext.gid.x) - CENTER_X;
  let dy = f32(activationContext.gid.y) - CENTER_Y;
  let distFactor = sin(sqrt(dx*dx + dy*dy) * RADIUS_SCALE + (activationContext.timestep / TIME_SCALE));

  return clamp(activationContext.cellState[activationContext.channel] + BASE_RATE * distFactor * tanh(x), 0.0, 1.0);
}`,

  "Perlin Influence": `// spatial frequency: maps pixel coords → noise space. smaller → broad/slow-changing blobs,
// larger → fine detail. (affects feature size)
const NOISE_SCALE: f32 = 150; @variable 10 500     
// temporal speed multiplier: multiplies activationContext.timestep when sampling noise.
// larger → faster animation; tune relative to your timestep units.
const TIME_SPEED: f32 = 500; @variable 1 1000     
// fBm octaves: number of noise layers summed. integer-valued (use whole numbers).
// more octaves → richer detail but higher cost.
const NOISE_OCTAVES: f32 = 8; @variable 1 8  
// amplitude falloff per octave: lower → smoother,
// higher → more high-frequency energy. typical: 0.3–0.7.
const PERSISTENCE: f32 = 0; @variable 0.0 1.0   
// frequency multiplier per octave: >1 increases frequency each octave.
// common: ~1.5–3.0 (higher → more rapid detail growth).
const LACUNARITY: f32 = 4; @variable 1.0 4.0 
// strength of noise effect on cell state: scales noise contribution 
// (multiplied by tanh(x) in activation). larger → stronger updates; watch clamping.
const BASE_RATE: f32 = 0.05; @variable 0.0 2.0   
// max influence of the noise
const MAX_INFLUENCE: f32 = 0.8; @variable 0.1 1.0

// --- helper functions ---
fn fade(t: vec3<f32>) -> vec3<f32> {
  return t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
}

fn lerp(a: f32, b: f32, t: f32) -> f32 {
  return a + t * (b - a);
}

// A deterministic scalar hash from a 3D position -> [0,1)
fn hash(p: vec3<f32>) -> f32 {
  // constants chosen for nice distribution
  return fract(sin(dot(p, vec3<f32>(127.1, 311.7, 74.7))) * 43758.5453123);
}

// Produce a pseudo-random normalized gradient vector at integer lattice point p
fn grad3(p: vec3<f32>) -> vec3<f32> {
  // create three correlated random scalars and remap to [-1,1]
  let r1 = hash(p);
  let r2 = hash(p + vec3<f32>(1.3, 7.1, 4.7));
  let r3 = hash(p + vec3<f32>(5.2, 2.8, 9.4));
  let g = vec3<f32>(r1 * 2.0 - 1.0, r2 * 2.0 - 1.0, r3 * 2.0 - 1.0);
  // normalize to approximate a gradient direction
  return normalize(g);
}

// Classic Perlin-style 3D gradient noise returning roughly in [-1, 1]
fn perlin3(p: vec3<f32>) -> f32 {
  let Pi = floor(p);
  let Pf = fract(p);
  let u = fade(Pf);

  // sample gradients at 8 cube corners and dot with offset vectors
  let g000 = grad3(Pi + vec3<f32>(0.0, 0.0, 0.0));
  let g100 = grad3(Pi + vec3<f32>(1.0, 0.0, 0.0));
  let g010 = grad3(Pi + vec3<f32>(0.0, 1.0, 0.0));
  let g110 = grad3(Pi + vec3<f32>(1.0, 1.0, 0.0));
  let g001 = grad3(Pi + vec3<f32>(0.0, 0.0, 1.0));
  let g101 = grad3(Pi + vec3<f32>(1.0, 0.0, 1.0));
  let g011 = grad3(Pi + vec3<f32>(0.0, 1.0, 1.0));
  let g111 = grad3(Pi + vec3<f32>(1.0, 1.0, 1.0));

  let d000 = dot(g000, Pf - vec3<f32>(0.0, 0.0, 0.0));
  let d100 = dot(g100, Pf - vec3<f32>(1.0, 0.0, 0.0));
  let d010 = dot(g010, Pf - vec3<f32>(0.0, 1.0, 0.0));
  let d110 = dot(g110, Pf - vec3<f32>(1.0, 1.0, 0.0));
  let d001 = dot(g001, Pf - vec3<f32>(0.0, 0.0, 1.0));
  let d101 = dot(g101, Pf - vec3<f32>(1.0, 0.0, 1.0));
  let d011 = dot(g011, Pf - vec3<f32>(0.0, 1.0, 1.0));
  let d111 = dot(g111, Pf - vec3<f32>(1.0, 1.0, 1.0));

  // trilinear (smooth) interpolation
  let nx00 = lerp(d000, d100, u.x);
  let nx10 = lerp(d010, d110, u.x);
  let nx01 = lerp(d001, d101, u.x);
  let nx11 = lerp(d011, d111, u.x);

  let nxy0 = lerp(nx00, nx10, u.y);
  let nxy1 = lerp(nx01, nx11, u.y);

  return lerp(nxy0, nxy1, u.z);
}

// Fractal Brownian Motion wrapper (fBm) using perlin3
fn perlinFBM(p: vec3<f32>) -> f32 {
  var amplitude: f32 = 1.0;
  var frequency: f32 = 1.0;
  var sum: f32 = 0.0;
  var maxAmp: f32 = 0.0;

  var i: u32 = 0u;
  loop {
    if (i >= u32(NOISE_OCTAVES)) { break; }
    sum = sum + perlin3(p * frequency) * amplitude;
    maxAmp = maxAmp + amplitude;
    amplitude = amplitude * PERSISTENCE;
    frequency = frequency * LACUNARITY;
    i = i + 1u;
  }

  // normalize to roughly [-1,1]
  return sum / maxAmp;
}

// --- Define Whatever function here using the noise ---
fn activation(x: f32) -> f32 {
  let px = f32(activationContext.gid.x) / NOISE_SCALE;
  let py = f32(activationContext.gid.y) / NOISE_SCALE;
  let time = activationContext.timestep / TIME_SPEED;

  // animated 3D point: (x, y, t)
  let raw = perlinFBM(vec3<f32>(px, py, time));

  // perlin3/perlinFBM returns roughly in [-1,1] already, but clamp/scale if needed
  let signedNoise = clamp(raw, -1.0, 1.0);

  // use signedNoise in place of your previous distFactor
  return clamp(
    activationContext.cellState[activationContext.channel] + BASE_RATE * (signedNoise * MAX_INFLUENCE) * tanh(x),
    0.0,
    1.0
  );
}`,

  "Brightness Switching Radial" : `const CENTER_X: f32 = f32(WIDTH) / 2.0;
const CENTER_Y: f32 = f32(HEIGHT) / 2.0;
const RADIUS_SCALE: f32 = 0.04; @variable 0.0 1.0
const BASE_RATE: f32 = -0.04; @variable -2.0 2.0
const BRIGHT_THRESHOLD: f32 = 0.5; @variable 0.0 1.0
const DARK_GROWTH: f32 = 0.04; @variable -0.5 0.5
const BRIGHT_DECAY: f32 = -0.05; @variable -0.5 0.5
const WAVE_INFLUENCE: f32 = 0.5; @variable 0.0 1.0

fn activation(x: f32) -> f32 {
    // Calculate radial wave pattern
    let dx = f32(activationContext.gid.x) - CENTER_X;
    let dy = f32(activationContext.gid.y) - CENTER_Y;
    let distFactor = sin(sqrt(dx*dx + dy*dy) * RADIUS_SCALE + (activationContext.timestep / 100));
    
    // Calculate brightness-based growth/decay
    let brightness = (activationContext.cellState.r +
       activationContext.cellState.g +
       activationContext.cellState.b) / 3.0;
    let brightnessRate = select(BRIGHT_DECAY, DARK_GROWTH, brightness < BRIGHT_THRESHOLD);
    
    // Combine both effects
    let waveEffect = BASE_RATE * distFactor;
    let combinedRate = mix(brightnessRate, waveEffect, WAVE_INFLUENCE);
    
    return clamp(activationContext.cellState[activationContext.channel] + combinedRate * tanh(x), 0.0, 1.0);
}`,

  "Exponential Linear Unit": `fn activation(x: f32) -> f32 {
  return select(exp(x) - 1.0, x, x >= 0.0);
}`,

  "Binary Spike Detector": `fn activation(x: f32) -> f32 {
  if (x == 3.0 || x == 11.0 || x == 12.0) {
    return 1.0;
  }
  return 0.0;
}`,

  "Inverse Gaussian": `fn activation(x: f32) -> f32 {
  return -1.0 / pow(2.0, 0.6 * x * x) + 1.0;
}`,

  "Tanh": `fn activation(x: f32) -> f32 {
  let e1 = exp(x);
  let e2 = exp(-x);
  return (e1 - e2) / (e1 + e2);
}`,

};
