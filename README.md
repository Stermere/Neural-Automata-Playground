# Neural Automata Playground (WebGPU)

A lightweight WebGPU-powered visualizer for experimenting with neural cellular automata.  
This project uses a GPU compute shader to apply 5×5 convolution filters over a color grid, allowing real-time updates and tunable parameters like per-channel weights and activation functions.

## Overview

- Each pixel is updated using a 5×5 convolution per color channel (R, G, B).
- You can modify **225 total weights** (3 output × 3 input × 5×5) in real time.
- Supports entering an activation function in WebGPU Shading Language.
- Built with WebGPU and React.

## Getting Started

### Prerequisites

- Node.js ≥ 18
- A browser and device with [WebGPU support](https://caniuse.com/webgpu)

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
