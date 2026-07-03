// Dev smoke test: drives the built app in Chrome and checks that the compute
// shader compiles and renders at 3 and 11 channels, and that share URLs round-trip.
// Usage: node scripts/smokeTest.mjs (expects `vite preview` running on :4173)
import puppeteer from 'puppeteer-core';

const CHROME = 'C:/Program Files/Google/Chrome/Application/chrome.exe';
const URL = 'http://localhost:4173/Neural-Automata-Playground/?width=256&height=256';

const errors = [];
const sleep = (ms) => new Promise(r => setTimeout(r, ms));

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: true,
  args: [
    '--enable-unsafe-webgpu',
    '--enable-features=Vulkan',
    '--no-first-run',
  ],
});

const page = await browser.newPage();
page.on('console', (msg) => {
  if (msg.type() === 'error' || msg.type() === 'warning') {
    errors.push(`[console.${msg.type()}] ${msg.text().slice(0, 2000)}`);
  }
});
page.on('pageerror', (err) => errors.push(`[pageerror] ${String(err).slice(0, 2000)}`));

const report = (label) => {
  console.log(`\n=== ${label} ===`);
  if (errors.length === 0) {
    console.log('no errors');
  } else {
    for (const e of errors) console.log(e);
    errors.length = 0;
  }
};

await page.goto(URL, { waitUntil: 'networkidle0' });
await sleep(2500);

const gpuInfo = await page.evaluate(async () => {
  if (!navigator.gpu) return 'navigator.gpu missing';
  const adapter = await navigator.gpu.requestAdapter();
  return adapter ? 'adapter ok' : 'no adapter';
});
console.log('WebGPU:', gpuInfo);

await page.screenshot({ path: 'scripts/smoke-3ch.png' });
report('initial load (3 channels, default config)');

// Open the Weight Editor tab and switch to 11 channels
const clicked = await page.evaluate(() => {
  const tab = [...document.querySelectorAll('button')].find(b => b.textContent?.trim() === 'Weight Editor');
  if (!tab) return 'no Weight Editor tab';
  tab.click();
  return 'ok';
});
console.log('open weight editor:', clicked);
await sleep(500);

const switched = await page.evaluate(() => {
  const select = document.querySelector('#channelCount');
  if (!select) return 'no #channelCount select';
  const setter = Object.getOwnPropertyDescriptor(HTMLSelectElement.prototype, 'value').set;
  setter.call(select, '11');
  select.dispatchEvent(new Event('change', { bubbles: true }));
  return 'ok';
});
console.log('switch to 11 channels:', switched);
await sleep(2500);

// Randomize so the hidden channels get non-zero state
await page.evaluate(() => {
  const btn = [...document.querySelectorAll('button')].find(b => /random/i.test(b.textContent ?? ''));
  btn?.click();
});
await sleep(2000);
await page.screenshot({ path: 'scripts/smoke-11ch.png' });
report('after switching to 11 channels + randomize');

// Share round-trip: click Share (clipboard may fail; replaceState still runs)
await page.evaluate(() => {
  const btn = [...document.querySelectorAll('button')].find(b => b.textContent?.trim() === 'Share');
  btn?.click();
});
await sleep(1500);
const sharedUrl = await page.evaluate(() => window.location.href);
console.log('\nshare url length:', sharedUrl.length, 'has #cfg=:', sharedUrl.includes('#cfg='));
report('after share click');

if (sharedUrl.includes('#cfg=')) {
  await page.goto(sharedUrl, { waitUntil: 'networkidle0' });
  await page.reload({ waitUntil: 'networkidle0' });
  await sleep(3000);
  const channelCount = await page.evaluate(() => {
    const tab = [...document.querySelectorAll('button')].find(b => b.textContent?.trim() === 'Weight Editor');
    tab?.click();
    return new Promise(resolve => setTimeout(() => {
      resolve(document.querySelector('#channelCount')?.value ?? 'select not found');
    }, 300));
  });
  console.log('\nchannel count after loading share url:', channelCount);
  await page.screenshot({ path: 'scripts/smoke-shared.png' });
  report('after loading share url');
}

await browser.close();
console.log('\ndone');
