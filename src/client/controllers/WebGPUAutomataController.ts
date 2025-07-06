import computeShaderCode from '../../shaders/compute.wgsl?raw';
import renderShaderCode from '../../shaders/render.wgsl?raw';
import { BASE_ACTIVATIONS } from '../constants/baseActivations';

const NORMALIZE_TRUE = 'let norm = x / max(weightSum, 1e-5);';
const NORMALIZE_FALSE = 'let norm = x;'

// src/controllers/WebGPUNeuralAutomataController.ts
export interface AutomataConfig {
  canvas: HTMLCanvasElement;
  gridSize: [number, number];
  brushRadius?: number;
  maxFps?: number;
  paused?: boolean;
}

export class WebGPUNeuralAutomataController {
  private device!: GPUDevice;
  private context!: GPUCanvasContext;
  private format: GPUTextureFormat = 'rgba8unorm';
  private sampler!: GPUSampler;

  private weightBuffer!: GPUBuffer;
  private texA!: GPUTexture;
  private texB!: GPUTexture;
  private bindA!: GPUBindGroup;
  private bindB!: GPUBindGroup;
  private renderBind!: GPUBindGroup;

  private computePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;

  private animationId = 0;
  private drawing = false;
  private brushRadius: number;

  private baseShaderCode = computeShaderCode;
  private activationCode = BASE_ACTIVATIONS.Linear;
  private normalizeInput = false;

  private maxFps: number;
  private frameInterval: number;
  private lastFrameTime = 0;
  private paused: boolean;

  constructor(private config: AutomataConfig) {
    this.brushRadius = config.brushRadius ?? 20;
    this.maxFps = config.maxFps ?? 240;
    this.frameInterval = 1000 / this.maxFps;
    this.paused = config.paused ?? false;
  }

  async init(): Promise<void> {
    const { canvas, gridSize } = this.config;
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter!.requestDevice();
    this.context = canvas.getContext('webgpu') as GPUCanvasContext;
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'premultiplied' });
    this.sampler = this.device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

    // Textures
    this.texA = this.createTexture(gridSize);
    this.texB = this.createTexture(gridSize);

    // Weights
    this.weightBuffer = this.createWeights();

    // Pipelines
    this.computePipeline = await this.createComputePipeline(this.buildComputeShaderCode());
    this.renderPipeline  = await this.createRenderPipeline(renderShaderCode);

    // Bind groups
    this.bindA = this.makeBindGroup(this.texA, this.texB);
    this.bindB = this.makeBindGroup(this.texB, this.texA);
    this.updateRenderBind();

    // Mouse events
    this.setupMouse(canvas, gridSize);

    // Start loop
    this.startLoop(gridSize);
  }

  updateWeights(flatWeights: number[]) {
    const buffer = new Float32Array(flatWeights);
    this.device.queue.writeBuffer(this.weightBuffer, 0, buffer);
  }

  setActivationFunction(update: { code: string; normalize: boolean }) {
    this.activationCode = update.code;
    this.normalizeInput = update.normalize;
    this.recompileComputePipeline();
  }

  clearCanvas(): void {
    const [w, h] = this.config.gridSize;
    const totalPixels = w * h;
    const pixels = new Uint8Array(totalPixels * 4);

    for (let i = 0; i < totalPixels; i++) {
      pixels[i * 4 + 0] = 0;   // R
      pixels[i * 4 + 1] = 0;   // G
      pixels[i * 4 + 2] = 0;   // B
      pixels[i * 4 + 3] = 255; // A
    }

    const layout = { bytesPerRow: w * 4 };
    const size = [w, h, 1];

    this.device.queue.writeTexture({ texture: this.texA, origin: [0, 0, 0] }, pixels, layout, size);
    this.device.queue.writeTexture({ texture: this.texB, origin: [0, 0, 0] }, pixels, layout, size);
  }

  randomizeCanvas(): void {
    const [w, h] = this.config.gridSize;
    const pixels = new Uint8Array(w * h * 4);
    for (let i = 0; i < w * h; i++) {
      pixels[i * 4 + 0] = Math.random() * 256;
      pixels[i * 4 + 1] = Math.random() * 256;
      pixels[i * 4 + 2] = Math.random() * 256;
      pixels[i * 4 + 3] = 255;
    }

    const layout = { bytesPerRow: w * 4 };
    const size = [w, h, 1];

    this.device.queue.writeTexture({ texture: this.texA, origin: [0, 0, 0] }, pixels, layout, size);
    this.device.queue.writeTexture({ texture: this.texB, origin: [0, 0, 0] }, pixels, layout, size);
  }

  setMaxFps(fps: number): void {
    this.maxFps = Math.max(1, fps); // Ensure minimum 1 FPS
    this.frameInterval = 1000 / this.maxFps;
  }

  togglePaused(paused: boolean): void {
    this.paused = paused;
  }

  private createWeights(): GPUBuffer {
    const outputChannels = 3;
    const inputChannels = 3;
    const kernelSize = 5;
    const weightsPerFilter = kernelSize * kernelSize;
    const totalWeights = outputChannels * inputChannels * weightsPerFilter;
    const weights = new Float32Array(totalWeights);

    for (let i = 0; i < totalWeights; i++) {
      weights[i] = 1.0;
    }

    const buffer = this.device.createBuffer({
      size: weights.byteLength,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
      mappedAtCreation: true,
    });

    new Float32Array(buffer.getMappedRange()).set(weights);
    buffer.unmap();

    this.weightBuffer = buffer;
    return buffer;
  }

  private createTexture(size: [number, number]): GPUTexture {
    return this.device.createTexture({
      size: [...size, 1], format: this.format,
      usage: GPUTextureUsage.TEXTURE_BINDING |
             GPUTextureUsage.STORAGE_BINDING |
             GPUTextureUsage.COPY_DST,
    });
  }

  private async createComputePipeline(code: string): Promise<GPUComputePipeline> {
    const module = this.device.createShaderModule({ code });
    return this.device.createComputePipeline({ layout: 'auto', compute: { module, entryPoint: 'main' } });
  }

  private async createRenderPipeline(code: string): Promise<GPURenderPipeline> {
    const module = this.device.createShaderModule({ code });
    return this.device.createRenderPipeline({
      layout: 'auto',
      vertex: { module, entryPoint: 'vs' },
      fragment: { module, entryPoint: 'fs', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-strip' },
    });
  }

  private async recompileComputePipeline() {
    this.computePipeline = await this.createComputePipeline(this.buildComputeShaderCode());
    this.bindA = this.makeBindGroup(this.texA, this.texB);
    this.bindB = this.makeBindGroup(this.texB, this.texA);
  }

  // fills in variable segments of the compute shader 
  private buildComputeShaderCode(): string {


    return this.baseShaderCode.replace(
      '@activationFunction',
      this.activationCode,
    ).replace(
      '@normalizeFlag',
      this.normalizeInput ? NORMALIZE_TRUE : NORMALIZE_FALSE,
    )
  }

  private makeBindGroup(src: GPUTexture, dst: GPUTexture): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: src.createView() },
        { binding: 1, resource: dst.createView() },
        { binding: 2, resource: { buffer: this.weightBuffer } },
      ],
    });
  }

  private updateRenderBind(): void {
    this.renderBind = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.texB.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
  }

  private setupMouse(canvas: HTMLCanvasElement, gridSize: [number, number]) {
    const getEventCoords = (event: MouseEvent | TouchEvent): [number, number] => {
      let clientX: number, clientY: number;

      if (event instanceof TouchEvent) {
        const touch = event.touches[0] || event.changedTouches[0];
        clientX = touch.clientX;
        clientY = touch.clientY;
      } else {
        clientX = event.clientX;
        clientY = event.clientY;
      }

      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;

      return this.canvasToGrid(x, y, canvas, gridSize);
    };

    const handleStart = (e: MouseEvent | TouchEvent) => {
      e.preventDefault?.();
      this.drawing = true;
      const [gx, gy] = getEventCoords(e);
      this.paintCell(gx, gy);
    };

    const handleMove = (e: MouseEvent | TouchEvent) => {
      if (!this.drawing) return;
      e.preventDefault?.();
      const [gx, gy] = getEventCoords(e);
      this.paintCell(gx, gy);
    };

    const handleEnd = () => {
      this.drawing = false;
    };

    // Mouse events
    canvas.addEventListener('mousedown', handleStart);
    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('mouseup', handleEnd);
    canvas.addEventListener('mouseleave', handleEnd);

    // Touch events
    canvas.addEventListener('touchstart', handleStart, { passive: false });
    canvas.addEventListener('touchmove', handleMove, { passive: false });
    canvas.addEventListener('touchend', handleEnd);
    canvas.addEventListener('touchcancel', handleEnd);
  }

  private canvasToGrid(x: number, y: number, canvas: HTMLCanvasElement, size: [number, number]): [number, number] {
    const rect = canvas.getBoundingClientRect();
    const gx = x * (size[0] / rect.width);
    const gy = y * (size[1] / rect.height);
    return [Math.floor(gx), Math.floor(gy)];
  }

  private paintCell(gx: number, gy: number): void {
    const [w, h] = this.config.gridSize;
    
    const minX = Math.max(0, gx - this.brushRadius);
    const maxX = Math.min(w - 1, gx + this.brushRadius);
    const minY = Math.max(0, gy - this.brushRadius);
    const maxY = Math.min(h - 1, gy + this.brushRadius);
    
    const width = maxX - minX + 1;
    const height = maxY - minY + 1;

    if (width <= 0 || height <= 0) {
      return;
    }
    
    const pixels = new Uint8Array(width * height * 4);
    
    for (let i = 0; i < width * height; i++) {
      pixels[i * 4 + 0] = 255; // R
      pixels[i * 4 + 1] = 255; // G
      pixels[i * 4 + 2] = 255; // B
      pixels[i * 4 + 3] = 255; // A
    }
    
    this.device.queue.writeTexture(
      { texture: this.texA, origin: [minX, minY, 0] },
      pixels,
      { bytesPerRow: width * 4, rowsPerImage: height },
      [width, height, 1]
    );
    this.device.queue.writeTexture(
      { texture: this.texB, origin: [minX, minY, 0] },
      pixels,
      { bytesPerRow: width * 4, rowsPerImage: height },
      [width, height, 1]
    );
  }

  private startLoop([w,h]: [number, number]) {
    const frame = (currentTime: number) => {
      // Check if enough time has passed since last frame
      if (currentTime - this.lastFrameTime < this.frameInterval) {
        this.animationId = requestAnimationFrame(frame);
        return;
      }
      this.lastFrameTime = currentTime;

      if (!this.paused) {
        // Compute
        const encC = this.device.createCommandEncoder();
        const passC = encC.beginComputePass();
        passC.setPipeline(this.computePipeline);
        passC.setBindGroup(0, this.bindA);
        passC.dispatchWorkgroups(w/16, h/16);
        passC.end();
        this.device.queue.submit([encC.finish()]);
      }

      // Render
      const encR = this.device.createCommandEncoder();
      const passR = encR.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp:'clear', storeOp:'store' }] });
      passR.setPipeline(this.renderPipeline);
      passR.setBindGroup(0, this.renderBind);
      passR.draw(4);
      passR.end();
      this.device.queue.submit([encR.finish()]);


      // Ping-pong
      if (!this.paused) {
        [this.texA, this.texB] = [this.texB, this.texA];
        [this.bindA, this.bindB] = [this.bindB, this.bindA];
        this.updateRenderBind();
      }

      this.animationId = requestAnimationFrame(frame);
    };

    this.animationId = requestAnimationFrame(frame);
  }

  destroy(): void {
    cancelAnimationFrame(this.animationId);
  }
}
