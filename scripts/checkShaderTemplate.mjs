// Dev sanity check: apply the same substitutions as buildComputeShaderCode and
// make sure no template tokens leak into the generated WGSL.
import { readFileSync } from 'fs';

const base = readFileSync(new URL('../src/shaders/compute.wgsl', import.meta.url), 'utf8');

const activation = `fn activation(x: f32) -> f32 {
  let u = activationContext.cellState.r;
  let v = activationContext.cellState.g;
  let t = activationContext.cellState.b;
  let all = activationContext.cellState.rgb;
  let self = activationContext.cellState[activationContext.channel];
  return select(exp(x) - 1.0, x, x >= 0.0) + (u + v + t + all.r + self) * 0.0;
}`;

const migrate = (code) => code
  .replace(
    /activationContext\.cellState\.rgb\b/g,
    '(vec3<f32>(activationContext.cellState[0], activationContext.cellState[1], activationContext.cellState[2]))',
  )
  .replace(
    /activationContext\.cellState\.([rgb])\b/g,
    (_m, ch) => `activationContext.cellState[${'rgb'.indexOf(ch)}]`,
  );

function build(channelCount) {
  return base
    .replace('@activationFunction', migrate(activation))
    .replace('@normalizeFlag', 'let norm = x;')
    .replace('@computeKernelFlag', 'var<private> COMPUTE_KERNEL: bool = true;')
    .replace('@sizeWidth', 1024)
    .replace('@sizeHeight', 1024)
    .replace('@channelCount', `${channelCount}`)
    .replace('@hiddenCount', `${channelCount - 3}`);
}

for (const count of [3, 11]) {
  const code = build(count);
  const leftovers = code.match(/@(?!group|binding|compute|workgroup_size|builtin|vertex|fragment|location|variable)\w+/g);
  console.log(`--- channelCount=${count} ---`);
  console.log('leftover template tokens:', leftovers ?? 'none');
  console.log('cellState swizzles remaining:', code.match(/cellState\.[rgb]/g) ?? 'none');
}

console.log('\n=== generated shader head (11 channels) ===');
console.log(build(11).split('\n').slice(0, 12).join('\n'));
console.log('\n=== migrated activation ===');
console.log(migrate(activation));
