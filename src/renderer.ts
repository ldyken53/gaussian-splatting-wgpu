import { PackedGaussians } from './ply';
import { f32, mat4x4 } from './packing';
import { RadixSorter } from './radix_sort';
import compute_depth from "./compute_depth.wgsl";
import compute_tiles from "./compute_tiles.wgsl";
import compute_ranges from "./compute_ranges.wgsl";
import render from "./render.wgsl";

const projMatrixLayout = new mat4x4(f32);

export class Renderer {
    canvas: HTMLCanvasElement;
    numGaussians: number;

    device: GPUDevice;
    contextGpu: GPUCanvasContext;

    radixSorter: RadixSorter;

    uniformBuffer: GPUBuffer;
    pointDataBuffer: GPUBuffer;
    drawIndexBuffer: GPUBuffer;
    projMatrixBuffer: GPUBuffer; // projection matrix, set at each frame
    depthBuffer: GPUBuffer; // depth values, computed each time using uniforms, padded to next power of 2
    indexBuffer: GPUBuffer; // buffer of gaussian indices (used for sort by depth)
    tileBuffer: GPUBuffer; // tile IDs for each gaussian
    rangesBuffer: GPUBuffer; // tile ranges for each pixel, pixel index written with stopping point in sorted gaussian buffer
    positionsBuffer: GPUBuffer;
    numGaussianBuffer: GPUBuffer;
    canvasSizeBuffer: GPUBuffer;
    tileSizeBuffer: GPUBuffer;
    numTilesBuffer: GPUBuffer;

    renderTarget: GPUTexture;

    renderPipelineBindGroup: GPUBindGroup;
    pointDataBindGroup: GPUBindGroup;
    computeDepthBindGroup: GPUBindGroup;
    computeTilesBindGroup: GPUBindGroup;
    computeRangesBindGroup: GPUBindGroup;

    renderPipeline: GPURenderPipeline;
    computeDepthPipeline: GPUComputePipeline;
    computeTilesPipeline: GPUComputePipeline;
    computeRangesPipeline: GPUComputePipeline;

    depthSortMatrix: number[][];

    // fps counter
    fpsCounter: HTMLLabelElement;
    lastDraw: number;

    destroyCallback: (() => void) | null = null;

    // destroy the renderer and return a promise that resolves when it's done (after the next frame)
    public async destroy(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.destroyCallback = resolve;
        });
    }

    constructor(
        canvas: HTMLCanvasElement,
        device: GPUDevice,
        gaussians: PackedGaussians,
    ) {
        this.canvas = canvas;
        this.device = device;
        const contextGpu = canvas.getContext("webgpu");
        if (!contextGpu) {
            throw new Error("WebGPU context not found!");
        }
        this.contextGpu = contextGpu;

        this.radixSorter = new RadixSorter(this.device);

        this.lastDraw = performance.now();

        this.numGaussians = gaussians.numGaussians;

        const presentationFormat = "rgba8unorm" as GPUTextureFormat;

        this.contextGpu.configure({
            device: this.device,
            format: presentationFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.pointDataBuffer = this.device.createBuffer({
            size: gaussians.gaussianArrayLayout.size,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
            label: "renderer.pointDataBuffer",
        });
        new Uint8Array(this.pointDataBuffer.getMappedRange()).set(new Uint8Array(gaussians.gaussiansBuffer));
        this.pointDataBuffer.unmap();

        // buffer for the vertex positions, set once
        this.positionsBuffer = this.device.createBuffer({
            size: gaussians.positionsArrayLayout.size,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
            label: "renderer.positionsBuffer"
        });
        new Uint8Array(this.positionsBuffer.getMappedRange()).set(new Uint8Array(gaussians.positionsBuffer));
        this.positionsBuffer.unmap();
        
        // buffer for the depth values, computed each time using uniforms, padded to next power of 2
        this.depthBuffer = this.device.createBuffer({
            size: this.radixSorter.getAlignedSize(this.numGaussians) * 4, // f32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            label: "renderer.depthBuffer"
        });

        // buffer for the index values, computed each time using uniforms, padded to next power of 2
        this.indexBuffer = this.device.createBuffer({
            size: this.radixSorter.getAlignedSize(this.numGaussians) * 4, // f32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.indexBuffer"
        });

        // buffer for the screen space xy values
        this.tileBuffer = this.device.createBuffer({
            size: this.radixSorter.getAlignedSize(this.numGaussians) * 4, // u32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.tileBuffer"
        });

        // buffer for the range of tiles for each pixel
        this.rangesBuffer = this.device.createBuffer({
            size: this.canvas.width  * this.canvas.height * 4 / (16 * 16), // u32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.rangesBuffer"
        });

        // buffer for the projection matrix, set at each frame
        this.projMatrixBuffer = this.device.createBuffer({
            size: projMatrixLayout.size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.projMatrixBuffer"
        });

        // buffer for the num gaussians, set once
        this.numGaussianBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.numGaussianBuffer"
        });
        this.device.queue.writeBuffer(
            this.numGaussianBuffer,
            0,
            new Uint32Array([this.numGaussians]),
            0,
            1
        );

        // buffer for the canvas size, set once
        this.canvasSizeBuffer = this.device.createBuffer({
            size: 2 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.canvasSizeBuffer"
        });
        this.device.queue.writeBuffer(
            this.canvasSizeBuffer,
            0,
            new Uint32Array([this.canvas.width, this.canvas.height]),
            0,
            2
        );

        // buffer for the tile size, set once, currently hardcoded
        this.tileSizeBuffer = this.device.createBuffer({
            size: 1 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.tileSizeBuffer"
        });
        this.device.queue.writeBuffer(
            this.tileSizeBuffer,
            0,
            new Uint32Array([16]),
            0,
            1
        );

        // buffer for the num tiles, set once, currently hardcoded
        this.numTilesBuffer = this.device.createBuffer({
            size: 1 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.numTilesBuffer"
        });
        this.device.queue.writeBuffer(
            this.numTilesBuffer,
            0,
            new Uint32Array([Math.ceil(this.canvas.width / 16) * Math.ceil(this.canvas.height / 16)]),
            0,
            1
        );

        this.computeDepthPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: compute_depth,
                }),
                entryPoint: "main",
            },
        });
        this.computeDepthBindGroup = this.device.createBindGroup({
            layout: this.computeDepthPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.positionsBuffer}},
                {binding: 1, resource: {buffer: this.depthBuffer}},
                {binding: 2, resource: {buffer: this.indexBuffer}},
                {binding: 3, resource: {buffer: this.tileBuffer}},
                {binding: 4, resource: {buffer: this.projMatrixBuffer}},
                {binding: 5, resource: {buffer: this.numGaussianBuffer}},
                {binding: 6, resource: {buffer: this.canvasSizeBuffer}},
                {binding: 7, resource: {buffer: this.tileSizeBuffer}}
            ]
        });

        this.renderTarget = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
                       GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST
        });
        this.computeTilesPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: compute_tiles,
                }),
                entryPoint: "main",
            },
        });
        this.computeTilesBindGroup = this.device.createBindGroup({
            layout: this.computeTilesPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: this.renderTarget.createView()},
                {binding: 1, resource: {buffer: this.rangesBuffer}},
                {binding: 2, resource: {buffer: this.canvasSizeBuffer}},
                {binding: 3, resource: {buffer: this.tileSizeBuffer}}
            ]
        });

        this.computeRangesPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: compute_ranges,
                }),
                entryPoint: "main",
            },
        });
        this.computeRangesBindGroup = this.device.createBindGroup({
            layout: this.computeRangesPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.tileBuffer}},
                {binding: 1, resource: {buffer: this.rangesBuffer}},
                {binding: 2, resource: {buffer: this.numTilesBuffer}},
            ]
        });

        let renderModule = this.device.createShaderModule({code: render});
        this.renderPipeline = this.device.createRenderPipeline({
            layout: "auto",
            vertex: {
                module: renderModule,
                entryPoint: "vertex_main",
            },
            fragment: {
                module: renderModule,
                entryPoint: "fragment_main",
                targets: [{format: presentationFormat}]
            },
        });
        const sampler = device.createSampler({
            magFilter: 'linear',
            minFilter: 'linear',
        });    
        this.renderPipelineBindGroup = device.createBindGroup({
            layout: this.renderPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: this.renderTarget.createView()},
                {binding: 1, resource: {buffer: this.canvasSizeBuffer}},
                {binding: 2, resource: sampler}
            ]
        });

        // start the animation loop
        requestAnimationFrame(() => this.animate());
    }

    private destroyImpl(): void {
        if (this.destroyCallback === null) {
            throw new Error("destroyImpl called without destroyCallback set!");
        }

        this.uniformBuffer.destroy();
        this.pointDataBuffer.destroy();
        this.drawIndexBuffer.destroy();
        this.destroyCallback();
    }

    async sort(projMatrix: number[][]) {
        const projMatrixCpuBuffer = new ArrayBuffer(projMatrixLayout.size);
        projMatrixLayout.pack(0, projMatrix, new DataView(projMatrixCpuBuffer));

        this.device.queue.writeBuffer(
            this.projMatrixBuffer,
            0,
            projMatrixCpuBuffer,
            0,
            projMatrixCpuBuffer.byteLength
        );

        { // compute the depth of each vertex
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computeDepthPipeline);
            passEncoder.setBindGroup(0, this.computeDepthBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.numGaussians / 64));
            passEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        // {
            // var dbgBuffer = this.device.createBuffer({
            //     size: this.tileBuffer.size,
            //     usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            // });

            // var commandEncoder = this.device.createCommandEncoder();
            // commandEncoder.copyBufferToBuffer(this.tileBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
            // this.device.queue.submit([commandEncoder.finish()]);
            // await this.device.queue.onSubmittedWorkDone();

            // await dbgBuffer.mapAsync(GPUMapMode.READ);

            // var debugVals = new Uint32Array(dbgBuffer.getMappedRange());
            // console.log(debugVals);
            // var minX = 0, maxX = 0, minY = 0, maxY = 0;
            // for (var i = 0; i < debugVals.length; i++) {
            //     if (i % 2 == 0) {
            //         if (debugVals[i] < minX) {
            //             minX = debugVals[i];
            //         }
            //         if (debugVals[i] > maxX) {
            //             maxX = debugVals[i];
            //         }
            //     } else {
            //         if (debugVals[i] < minY) {
            //             minY = debugVals[i];
            //         }
            //         if (debugVals[i] > maxY) {
            //             maxY = debugVals[i];
            //         }
            //     }
            // }
            // console.log(minX, maxX);
            // console.log(minY, maxY);
        // }
        await this.radixSorter.sort(this.tileBuffer, this.indexBuffer, this.numGaussians, false, false);
        // {
        //     var dbgBuffer = this.device.createBuffer({
        //         size: this.indexBuffer.size,
        //         usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        //     });

        //     var commandEncoder = this.device.createCommandEncoder();
        //     commandEncoder.copyBufferToBuffer(this.indexBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
        //     this.device.queue.submit([commandEncoder.finish()]);
        //     await this.device.queue.onSubmittedWorkDone();

        //     await dbgBuffer.mapAsync(GPUMapMode.READ);

        //     var debugVals2 = new Uint32Array(dbgBuffer.getMappedRange());
        //     console.log(debugVals2);
        // }
        {
            var dbgBuffer = this.device.createBuffer({
                size: this.tileBuffer.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            var commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(this.tileBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await dbgBuffer.mapAsync(GPUMapMode.READ);

            var debugVals = new Uint32Array(dbgBuffer.getMappedRange());
            console.log(debugVals);
        }
    }

    async animate() {
        console.log(this);
        if (this.destroyCallback !== null) {
            this.destroyImpl();
            return;
        }
        let defaultCam = [
            [0.9640601277351379, 0.021361779421567917, 0.2648240327835083, -0],
            [-0.0728282555937767, 0.9798307418823242, 0.18608540296554565, 0],
            [-0.25550761818885803, -0.1986841857433319, 0.9461714625358582, 0],
            [-0.8031625151634216, 0.6299861669540405, 5.257271766662598, 1]
        ]
        await this.sort(defaultCam);

        { 
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computeRangesPipeline);
            passEncoder.setBindGroup(0, this.computeRangesBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.numGaussians / 64));
            passEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        { 
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computeTilesPipeline);
            passEncoder.setBindGroup(0, this.computeTilesBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.canvas.width / 16), Math.ceil(this.canvas.height / 16));
            passEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);
        }

        {
            var dbgBuffer = this.device.createBuffer({
                size: this.rangesBuffer.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            var commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(this.rangesBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await dbgBuffer.mapAsync(GPUMapMode.READ);

            var debugVals = new Uint32Array(dbgBuffer.getMappedRange());
            console.log(debugVals);
        }

        { 
            // Blit the image rendered onto the screen
            const commandEncoder = this.device.createCommandEncoder();

            const renderPassDesc = {
                colorAttachments: [{
                    view: this.contextGpu.getCurrentTexture().createView(),
                    loadOp: "clear" as GPULoadOp,
                    clearValue: [0.3, 0.3, 0.3, 1],
                    storeOp: "store" as GPUStoreOp
                }],
            };
            const renderPass = commandEncoder.beginRenderPass(renderPassDesc);

            renderPass.setPipeline(this.renderPipeline);
            renderPass.setBindGroup(0, this.renderPipelineBindGroup);

            // Draw a full screen quad
            renderPass.draw(6, 1, 0, 0);
            renderPass.end();
            this.device.queue.submit([commandEncoder.finish()]);
        }

        // requestAnimationFrame(() => this.animate());
    }
}