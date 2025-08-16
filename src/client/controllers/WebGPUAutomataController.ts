import computeShaderCode from '../../shaders/compute.wgsl?raw';
import renderShaderCode from '../../shaders/render.wgsl?raw';
import { BASE_ACTIVATIONS } from '../constants/baseActivations';

const NORMALIZE_TRUE = 'let norm = x / max(weightSum, 1e-5);';
const NORMALIZE_FALSE = 'let norm = x;'
const COMPUTE_KERNEL_TRUE = 'var<private> COMPUTE_KERNEL: bool = true;';
const COMPUTE_KERNEL_FALSE = 'var<private> COMPUTE_KERNEL: bool = false;';

export interface AutomataConfig {
  canvas: HTMLCanvasElement;
  gridSize: [number, number];
  brushRadius?: number;
  maxFps?: number;
  paused?: boolean;
  isDraggable?: boolean;
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
  private renderBindA!: GPUBindGroup;
  private renderBindB!: GPUBindGroup;
  private renderBind!: GPUBindGroup;
  private renderBindATexture: GPUTexture | null = null;
  private renderBindBTexture: GPUTexture | null = null;

  private computePipeline!: GPUComputePipeline;
  private renderPipeline!: GPURenderPipeline;

  private gridSize: [number, number];
  private animationId = 0;
  private drawing = false;
  private brushRadius: number;

  private baseShaderCode = computeShaderCode;
  private activationCode = BASE_ACTIVATIONS["Exponential Linear Unit"];
  private normalizeInput = false;
  private computeKernel = true;

  private maxFps: number;
  private frameInterval: number;
  private lastFrameTime = 0;
  private paused: boolean;

  private isDragging = false;
  private dragStart = { x: 0, y: 0 };
  private panOffset = { x: 0, y: 0 };

  private timestep: number = 0;
  private timestepBuffer!: GPUBuffer;
  private lastClickX: number = 0;
  private lastClickY: number = 0;

  private isDraggable = true;

  private brushPixelBuffer: Uint8Array | null = null;
  private brushBufferWidth = 0;
  private brushBufferHeight = 0;

  constructor(private config: AutomataConfig) {
    this.brushRadius = config.brushRadius ?? 20;
    this.maxFps = config.maxFps ?? 240;
    this.frameInterval = 1000 / this.maxFps;
    this.paused = config.paused ?? false;
    this.gridSize = config.gridSize;
    this.isDraggable = config.isDraggable ?? true;
  }

  async init(): Promise<void> {
    const { canvas } = this.config;
    if (!navigator.gpu) throw new Error('WebGPU not supported');
    const adapter = await navigator.gpu.requestAdapter();
    this.device = await adapter!.requestDevice();
    this.context = canvas.getContext('webgpu') as GPUCanvasContext;
    this.context.configure({ device: this.device, format: this.format, alphaMode: 'premultiplied' });
    this.sampler = this.device.createSampler({ magFilter: 'nearest', minFilter: 'nearest' });

    // Textures
    this.texA = this.createTexture(this.gridSize);
    this.texB = this.createTexture(this.gridSize);

    // Weights
    this.weightBuffer = this.createWeights();

    // Timestep + Click data
    this.timestepBuffer = this.createTimestepBuffer();

    // Pipelines
    this.computePipeline = await this.createComputePipeline(this.buildComputeShaderCode());
    this.renderPipeline  = await this.createRenderPipeline(renderShaderCode);

    // Bind groups
    this.bindA = this.makeBindGroup(this.texA, this.texB);
    this.bindB = this.makeBindGroup(this.texB, this.texA);

    // Precreate render bind groups for both textures
    this.renderBindA = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.texA.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
    this.renderBindB = this.device.createBindGroup({
      layout: this.renderPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: this.texB.createView() },
        { binding: 1, resource: this.sampler },
      ],
    });
    this.renderBindATexture = this.texA;
    this.renderBindBTexture = this.texB;
    this.renderBind = this.renderBindB;

    // Mouse events
    this.setupMouse(canvas, this.gridSize);

    // Start loop
    this.startLoop(this.gridSize);
  }

  updateWeights(flatWeights: number[]) {
    const buffer = new Float32Array(flatWeights);
    this.device.queue.writeBuffer(this.weightBuffer, 0, buffer);
  }

  setActivationParameters(normalize: boolean, computeKernel?: boolean) {
    this.normalizeInput = normalize;
    this.computeKernel = computeKernel ?? this.computeKernel;
  }

  setActivationFunctionCode(code: string) {
    this.activationCode = code;
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
    this.timestep = 0;
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
    this.timestep = 0;
  }

  paintBoarders(): void {
    const [w, h] = this.gridSize;
    const total = w * h;
    const pixels = new Uint8Array(total * 4);
    const borderThickness = this.brushRadius; // Use brush radius for thickness

    // Fill background black (RGB=0) with alpha=255
    for (let i = 0; i < total; i++) {
      pixels[i * 4 + 0] = 0;
      pixels[i * 4 + 1] = 0;
      pixels[i * 4 + 2] = 0;
      pixels[i * 4 + 3] = 255;
    }

    // Fill border pixels white (RGB=255)
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const onTopEdge = y < borderThickness;
        const onBottomEdge = y >= h - borderThickness;
        const onLeftEdge = x < borderThickness;
        const onRightEdge = x >= w - borderThickness;

        if (onTopEdge || onBottomEdge || onLeftEdge || onRightEdge) {
          const idx = (y * w + x) * 4;
          pixels[idx + 0] = 255;
          pixels[idx + 1] = 255;
          pixels[idx + 2] = 255;
          pixels[idx + 3] = 255;
        }
      }
    }

    const layout = { bytesPerRow: w * 4 };
    const size = [w, h, 1];

    this.device.queue.writeTexture({ texture: this.texA, origin: [0, 0, 0] }, pixels, layout, size);
    this.device.queue.writeTexture({ texture: this.texB, origin: [0, 0, 0] }, pixels, layout, size);
  }

  setMaxFps(fps: number): void {
    this.maxFps = Math.max(1, fps);
    this.frameInterval = 1000 / this.maxFps;
  }

  setBrushSize(size: number): void {
    this.brushRadius = size;
  }

  togglePaused(paused: boolean): void {
    this.paused = paused;
  }

  setZoom(zoomLevel: number, transformTranslation?: string): void {
    const canvas = this.config.canvas;
    if (canvas) {
      if (transformTranslation) {
        const currentTransform = canvas.style.transform || '';
        const transformWithoutTranslation = currentTransform.replace(/translate\([^)]*\)/g, '');
        canvas.style.transform = `${transformWithoutTranslation} ${transformTranslation}`.trim();
      }

      const currentTransform = canvas.style.transform || '';
      const transformWithoutScale = currentTransform.replace(/scale\([^)]*\)/g, '');
      canvas.style.transform = `${transformWithoutScale} scale(${zoomLevel})`.trim();
    }
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

  private createTimestepBuffer(): GPUBuffer {
    const buffer = this.device.createBuffer({
      size: 4 * 4, // timestep, clickX, clickY, unused
      usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
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

  // Fills in variable segments of the compute shader 
  private buildComputeShaderCode(): string {
    return this.baseShaderCode.replace(
      '@activationFunction',
      this.activationCode,
    ).replace(
      '@normalizeFlag',
      this.normalizeInput ? NORMALIZE_TRUE : NORMALIZE_FALSE,
    ).replace(
      '@computeKernelFlag',
      this.computeKernel ? COMPUTE_KERNEL_TRUE : COMPUTE_KERNEL_FALSE,
    ).replace(
      '@sizeWidth',
      this.gridSize[0],
    ).replace(
      '@sizeHeight',
      this.gridSize[1],
    );
  }

  private makeBindGroup(src: GPUTexture, dst: GPUTexture): GPUBindGroup {
    return this.device.createBindGroup({
      layout: this.computePipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: src.createView() },
        { binding: 1, resource: dst.createView() },
        { binding: 2, resource: { buffer: this.weightBuffer } },
        { binding: 3, resource: { buffer: this.timestepBuffer } },
      ],
    });
  }

  private updateRenderBind(): void {
    if (this.renderBindATexture === this.texB) {
      this.renderBind = this.renderBindA;
      return;
    }
    if (this.renderBindBTexture === this.texB) {
      this.renderBind = this.renderBindB;
      return;
    }
  }

  // Setup panning and drawing functionality
  private setupMouse(canvas: HTMLCanvasElement, gridSize: [number, number]) {
    const isTouchEvent = (ev: any): ev is TouchEvent => !!ev && 'touches' in ev;
    const isMouseEvent = (ev: any): ev is MouseEvent => !!ev && typeof ev.clientX === 'number' && typeof ev.clientY === 'number';

    const getEventCoords = (event: MouseEvent | TouchEvent): [number, number] => {
      let clientX: number, clientY: number;
      if (isTouchEvent(event)) {
        const touch = (event as TouchEvent).touches?.[0] || (event as TouchEvent).changedTouches?.[0];
        clientX = touch.clientX;
        clientY = touch.clientY;
      } else {
        clientX = (event as MouseEvent).clientX;
        clientY = (event as MouseEvent).clientY;
      }

      const rect = canvas.getBoundingClientRect();
      const x = clientX - rect.left;
      const y = clientY - rect.top;

      return this.canvasToGrid(x, y, canvas, gridSize);
    };

    const handleStart = (e: MouseEvent | TouchEvent) => {
      if (isMouseEvent(e) && (e as MouseEvent).button === 2) {
        e.preventDefault();
        const currentTransform = canvas.style.transform;
        const translateMatch = currentTransform.match(/translate\(([^)]+)\)/);
        const translateValues = translateMatch?.[1].split(',') ?? ['0', '0'];
        this.panOffset.x = parseFloat(translateValues[0]) || 0;
        this.panOffset.y = parseFloat(translateValues[1]) || 0;
        this.isDragging = true;
        this.dragStart = { x: (e as MouseEvent).clientX - this.panOffset.x, y: (e as MouseEvent).clientY - this.panOffset.y };
        canvas.style.cursor = 'grabbing';
        return;
      }
      // call preventDefault if available (TouchEvent may not be present in some browsers)
      try { (e as any).preventDefault && (e as any).preventDefault(); } catch {}
      this.drawing = true;
      const [gx, gy] = getEventCoords(e);

      this.lastClickX = gx;
      this.lastClickY = gy;
      this.paintCell(gx, gy);
    };

    const handleMove = (e: MouseEvent | TouchEvent) => {
      if (isMouseEvent(e) && this.isDragging && this.isDraggable) {
        const newPanX = (e as MouseEvent).clientX - this.dragStart.x;
        const newPanY = (e as MouseEvent).clientY - this.dragStart.y;
        this.panOffset = { x: newPanX, y: newPanY };

        const currentTransform = canvas.style.transform;
        const scaleMatch = currentTransform.match(/scale\(([^)]+)\)/);
        const currentScale = scaleMatch ? scaleMatch[1] : '1';

        canvas.style.transform = `translate(${newPanX}px, ${newPanY}px) scale(${currentScale})`;
        return
      }

      if (!this.drawing) return;
      try { (e as any).preventDefault && (e as any).preventDefault(); } catch {}
      const [gx, gy] = getEventCoords(e);
      
      this.lastClickX = gx;
      this.lastClickY = gy;
      this.paintCell(gx, gy);
    };

    const handleEnd = () => {
      canvas.style.cursor = 'default';
      this.isDragging = false;
      this.drawing = false;
    };

    const handleContextMenu = (e: MouseEvent) => {
      e.preventDefault();
    };

    // Mouse and touch events 
    canvas.addEventListener('mousedown', handleStart);
    canvas.addEventListener('mousemove', handleMove);
    canvas.addEventListener('mouseup', handleEnd);
    canvas.addEventListener('contextmenu', handleContextMenu);
    canvas.addEventListener('touchstart', handleStart, { passive: false });
    canvas.addEventListener('touchmove', handleMove, { passive: false });
    canvas.addEventListener('touchend', handleEnd);
    canvas.addEventListener('touchcancel', handleEnd);

    window.addEventListener('mouseup', (e) => {
      if (this.drawing) {
        this.drawing = false;
        canvas.style.cursor = 'default';
      }
      if (this.isDragging) {
        this.isDragging = false;
        canvas.style.cursor = 'default';
      }
    });
  }

  private canvasToGrid(x: number, y: number, canvas: HTMLCanvasElement, size: [number, number]): [number, number] {
    const rect = canvas.getBoundingClientRect();
    const gx = x * (size[0] / rect.width);
    const gy = y * (size[1] / rect.height);
    return [Math.floor(gx), Math.floor(gy)];
  }

  private paintCell(gx: number, gy: number): void {
    if (!this.device || !this.texA || !this.texB) {
      return
    }

    const [w, h] = this.gridSize;
    
    const minX = Math.max(0, gx - this.brushRadius);
    const maxX = Math.min(w - 1, gx + this.brushRadius);
    const minY = Math.max(0, gy - this.brushRadius);
    const maxY = Math.min(h - 1, gy + this.brushRadius);
    
    const width = maxX - minX + 1;
    const height = maxY - minY + 1;

    if (width <= 0 || height <= 0) {
      return;
    }
    
    // Reuse a preallocated pixel buffer when possible
    const neededSize = width * height * 4;
    if (!this.brushPixelBuffer || this.brushBufferWidth !== width || this.brushBufferHeight !== height) {
      this.brushPixelBuffer = new Uint8Array(neededSize);
      this.brushBufferWidth = width;
      this.brushBufferHeight = height;
    }
    const pixels = this.brushPixelBuffer;
    pixels.fill(255);
    
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
      if (currentTime - this.lastFrameTime < this.frameInterval) {
        this.animationId = requestAnimationFrame(frame);
        return;
      }
      this.lastFrameTime = currentTime;

      // Single command encoder for both compute and render to reduce submit overhead
      const encoder = this.device.createCommandEncoder();

      if (!this.paused) {
        const timestepData = new Float32Array([
          this.timestep,
          this.lastClickX,
          this.lastClickY,
          0.0 // Allignment
        ]);
        this.device.queue.writeBuffer(this.timestepBuffer, 0, timestepData);
        this.timestep += 1;

        // Compute pass
        const computePass = encoder.beginComputePass();
        computePass.setPipeline(this.computePipeline);
        computePass.setBindGroup(0, this.bindA);
        computePass.dispatchWorkgroups(Math.ceil(w / 16), Math.ceil(h / 16));
        computePass.end();
      }

      // Render pass (always run to present the current texture)
      const renderPass = encoder.beginRenderPass({ colorAttachments: [{ view: this.context.getCurrentTexture().createView(), loadOp:'clear', storeOp:'store' }] });
      renderPass.setPipeline(this.renderPipeline);
      renderPass.setBindGroup(0, this.renderBind);
      renderPass.draw(4);
      renderPass.end();

      // Submit once
      this.device.queue.submit([encoder.finish()]);

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
    // Cancel animation frame
    if (this.animationId) {
      cancelAnimationFrame(this.animationId);
      this.animationId = 0;
    }

    // Destroy GPU buffers
    if (this.weightBuffer) {
      this.weightBuffer.destroy();
      this.weightBuffer = undefined as any;
    }
    if (this.timestepBuffer) {
      this.timestepBuffer.destroy();
      this.timestepBuffer = undefined as any;
    }

    // Destroy textures
    if (this.texA) {
      this.texA.destroy();
      this.texA = undefined as any;
    }
    if (this.texB) {
      this.texB.destroy();
      this.texB = undefined as any;
    }

    // Clear other references
    this.bindA = undefined as any;
    this.bindB = undefined as any;
    this.renderBindA = undefined as any;
    this.renderBindB = undefined as any;
    this.renderBind = undefined as any;
    this.computePipeline = undefined as any;
    this.renderPipeline = undefined as any;
  }
}
