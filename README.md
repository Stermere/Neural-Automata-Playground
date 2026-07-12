# Neural Automata Playground (WebGPU)

A lightweight WebGPU-powered visualizer for experimenting with neural cellular automata.  
This project uses a GPU compute shader to apply 5×5 convolution filters over a grid of pixels, allowing real-time updates to the weights and activation functions that control the behavior of the automata.

**Hosted on GitHub Pages [here](https://stermere.github.io/Neural-Automata-Playground/)**

<details open>
<summary>🌀 Show Rings</summary>

<img src="readMeAssets/Rings.gif" alt="Rings" width="90%"/>

</details>

<details>
<summary>💜 Show Purple Goo</summary>

<img src="readMeAssets/PurpleGoo.gif" alt="Purple Goo" width="90%"/>

</details>

<details>
<summary>🌊 Show Neon Wave</summary>

<img src="readMeAssets/NeonWave.gif" alt="Neon Wave" width="90%"/>

</details>

<details>
<summary>🐛 Show Big Worms</summary>

<img src="readMeAssets/BigWorms.gif" alt="Big Worms" width="90%"/>

</details>

## Overview

- Each pixel is updated using a 5×5 convolution over up to 16 channels — 3 visible (R, G, B) plus up to 13 hidden memory channels.
- You can modify every kernel weight in real time (225 for a plain RGB config, more as channels are added).
- Supports a nearly unlimited variety of activation functions, if you can write it in WGSL you can use it! 
- Pre-trained patterns (see `kernalPreTraining/`) can add a per-cell MLP between the convolution and the activation, for update rules a single conv can't express — the conv weights stay editable either way.
- Built with WebGPU and React.

## Getting Started

### Prerequisites

- Node.js ≥ 18
- A browser and device with [WebGPU support](https://caniuse.com/webgpu). WebGPU is not enabled by default on some browsers 

### Installation

```bash
git clone https://github.com/Stermere/Neural-Automata-Playground/
cd Neural-Automata-Playground
npm install
```

### Local development server

```bash
# Add -- --host to host it on your local network
npm run dev
```

### URL options

The playground accepts these optional query parameters:

- `width` / `height`: canvas dimensions (default: `1024`).
- `brushSize`: initial brush size from `1` to `100` pixels (default: `20`).
- `initialState`: canvas contents on load. Use `random` (default), `dot` for one white pixel in the center, or `none` for an empty black canvas.

For example:

```text
?width=512&height=512&brushSize=1&initialState=dot
```
