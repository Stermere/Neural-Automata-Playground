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
  
  "Channel Cross Talk": `const COUPLING: f32 = 0.5; @variable -1.0 1.0
const BASE_RATE: f32 = 0.0; @variable -2.0 2.0
const COUPLING_CENTER: f32 = 0.5; @variable 0.0 1.0

fn activation(x: f32) -> f32 {
  let r = activationContext.cellState.r;
  let g = activationContext.cellState.g;
  let b = activationContext.cellState.b;
  var coupledVal: f32;
  if (activationContext.channel == 0u) { // red depends on green
      coupledVal = g;
  } else if (activationContext.channel == 1u) { // green depends on blue
      coupledVal = b;
  } else { // blue depends on red
      coupledVal = r;
  }
  return clamp(
      (activationContext.cellState[activationContext.channel] + 
        x * (BASE_RATE + COUPLING * (coupledVal - COUPLING_CENTER))), 
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
