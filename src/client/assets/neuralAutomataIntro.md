# Neural Cellular Automata
Neural Cellular Automata extend classic automata with the expressive power of convolutional neural networks allowing each pixel to update itself based on its neighbors using tunable, rule-based logic. Each pixel derives its state by applying a tunable 5x5 kernel function, enabling complex, lifelike patterns and self-organizing behavior. 

You can think of each pixel as a small, independent unit that reacts to its surroundings based on a shared update rule. When thousands of these units operate together, they produce dynamic patterns that shift, stabilize, grow, or decay often in surprising and visually striking ways.

**Warning:** You will inevitably create bright flashing patterns with this tool. You have been warned!

---

## Features

### Weight Editor
- **Visual weight grid**: See and tweak the 5×5 convolutional kernels that determine how neighboring cells influence one another.
- **Per‑channel control**: Edit separate kernels for each channel (R, G, B) to craft multi‑channel interactions.
- **Live updates**: Changes to any weight immediately take effect in the simulation.

### Activation Editor
- **WGSL code editor**: Write or paste custom activation functions in WebGPU Shading Language.
- **Preset library**: Choose from built‑in activations like Linear, ELU, Softsign, Swish, Mish, Inverse Gaussian, and more.
- **Live updates**: Instantly compile and swap in new activation logic without reloading the page.

### General Features
- **WebGPU acceleration**: All cellular updates run as compute shaders for high performance and scalability.
- **Brush controls**: Paint initial conditions or perturb existing patterns.
- **Save & load**: Store and recall any combination of weights + activation in your browser’s localStorage.
- **Import/export**: Download your configurations as JSON files or upload your own to share and collaborate.

---
## Activation Context
This construct is globally scoped and available for use within the activation function enabling more complex activation functions.
```bash
struct ActivationContext {
    gid: vec3<u32>,
    weightSum: f32,
    cellState: vec3<f32>,
    channel: u32
}

var<private> activationContext: ActivationContext;

fn activation(x: f32) -> f32 {
    return x;
}
```
- **gid**: This is the pixel being processed, gid.x and gid.y tell you the coordinates of the current pixel.
- **weightSum**: This is the sum of all the weights while calculating x.
- **cellState**: this is the pixel values of the current gid, cellState.r yields the red value of the pixel.
- **channel**: This is the channel whose value is being calculated via the current call to activationFunction. Channel takes on the values 0, 1, and 2 corresponding to r, g, and b.

---


## The Math

Some details are omitted for simplicity, you can always view the source code [here](https://github.com/Stermere/Neural-Automata-Playground/) if you're interested in the fine-grained details.

- The input image is a matrix with size 3x1024x1024
- Output image is a matrix with the same size as the Input image
- Kernel is a matrix with size 3x3x5x5
- Activation function is any function that takes a number
