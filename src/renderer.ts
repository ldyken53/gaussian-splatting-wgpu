import { Mat4 } from 'wgpu-matrix';

import { PackedGaussians } from './ply';
import { Struct, f32, mat4x4, vec3 } from './packing';
import { ExclusiveScanPipeline, ExclusiveScanner } from './exclusive_scan';
import { InteractiveCamera } from './camera';

import process_tetras from "./process_tetras.wgsl";
import compute_tiles from "./compute_tiles.wgsl";
import compute_ranges from "./compute_ranges.wgsl";
import render from "./render.wgsl";
import write_tile_ids from "./write_tile_ids.wgsl";
import { GPUSorter } from './radix_sort/sort';
import { Volume } from './off';

const uniformLayout = new Struct([
    ['viewMatrix', new mat4x4(f32)],
    ['projMatrix', new mat4x4(f32)],
    ['cameraPosition', new vec3(f32)],
    ['tanHalfFovX', f32],
    ['tanHalfFovY', f32],
    ['focalX', f32],
    ['focalY', f32],
    ['scaleModifier', f32],
]);

function mat4toArrayOfArrays(m: Mat4): number[][] {
    return [
        [m[0], m[1], m[2], m[3]],
        [m[4], m[5], m[6], m[7]],
        [m[8], m[9], m[10], m[11]],
        [m[12], m[13], m[14], m[15]],
    ];
}

export class Renderer {
    canvas: HTMLCanvasElement;
    interactiveCamera: InteractiveCamera;

    numPoints: number;
    numTetra: number;
    tileSize: number;
    numIntersections: number;
    numFrames: number;

    device: GPUDevice;
    contextGpu: GPUCanvasContext;

    sorter: GPUSorter;
    scanPipeline: ExclusiveScanPipeline;
    scanTileCounts: ExclusiveScanner;

    uniformBuffer: GPUBuffer; // camera uniforms
    pointDataBuffer: GPUBuffer;
    tetraIDBuffer: GPUBuffer; // buffer of tetra indices (used for sort by tile and depth)
    tileCountBuffer: GPUBuffer; // used to count the number of tile intersections for each Gaussian
    tileOffsetBuffer: GPUBuffer; // filled with output of prefix sum on tileCountBuffer
    tileIDBuffer: GPUBuffer; // tile IDs for each gaussian
    rangesBuffer: GPUBuffer; // tile ranges for each pixel, pixel index written with stopping point in sorted gaussian buffer

    canvasSizeBuffer: GPUBuffer;
    tileSizeBuffer: GPUBuffer;
    numTilesBuffer: GPUBuffer;

    renderTarget: GPUTexture;
    renderTargetCopy: GPUTexture;

    renderPipelineBindGroup: GPUBindGroup;
    pointDataBindGroup: GPUBindGroup;
    writeTileIDsBindGroup: GPUBindGroup;
    computeTilesBindGroup: GPUBindGroup;
    computeRangesBindGroup: GPUBindGroup;

    renderPipeline: GPURenderPipeline;
    writeTileIDsPipeline: GPUComputePipeline;
    computeTilesPipeline: GPUComputePipeline;
    computeRangesPipeline: GPUComputePipeline;

    depthSortMatrix: number[][];

    // fps counter
    fpsCounter: HTMLLabelElement;
    lastDraw: number;

    destroyCallback: (() => void) | null = null;
    tetraDataBuffer: GPUBuffer;
    numTetraBuffer: GPUBuffer;
    tetraDepthBuffer: GPUBuffer;
    tetraRectBuffer: GPUBuffer;
    processTetraPipeline: GPUComputePipeline;
    processTetraBindGroup: GPUBindGroup;
    tetraUVBuffer: GPUBuffer;
    visibleFacesBuffer: GPUBuffer;

    // destroy the renderer and return a promise that resolves when it's done (after the next frame)
    public async destroy(): Promise<void> {
        return new Promise((resolve, reject) => {
            this.destroyCallback = resolve;
        });
    }

    constructor(
        canvas: HTMLCanvasElement,
        interactiveCamera: InteractiveCamera,
        device: GPUDevice,
        volume: Volume,
        tileSize: number
    ) {
        this.tileSize = tileSize;
        this.canvas = canvas;
        this.interactiveCamera = interactiveCamera;
        this.device = device;
        this.numFrames = 0;
        const contextGpu = canvas.getContext("webgpu");
        if (!contextGpu) {
            throw new Error("WebGPU context not found!");
        }
        this.contextGpu = contextGpu;

        this.sorter = new GPUSorter(this.device, 32);
        this.scanPipeline = new ExclusiveScanPipeline(this.device);

        this.lastDraw = performance.now();

        this.numPoints = volume.numPoints;
        this.numTetra = volume.numTetra;
        console.log(`Num points: ${this.numPoints}, num tetra: ${this.numTetra}`);

        const presentationFormat = "rgba8unorm" as GPUTextureFormat;

        this.contextGpu.configure({
            device: this.device,
            format: presentationFormat,
            usage: GPUTextureUsage.RENDER_ATTACHMENT
        });

        this.pointDataBuffer = this.device.createBuffer({
            size: volume.points.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
            label: "renderer.pointDataBuffer",
        });
        new Float32Array(this.pointDataBuffer.getMappedRange()).set(volume.points);
        this.pointDataBuffer.unmap();

        this.tetraDataBuffer = this.device.createBuffer({
            size: volume.tetra.byteLength,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
            label: "renderer.tetraDataBuffer",
        });
        new Uint32Array(this.tetraDataBuffer.getMappedRange()).set(volume.tetra);
        this.tetraDataBuffer.unmap();

        this.tetraRectBuffer = this.device.createBuffer({
            size: this.numTetra * 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.tetraRectBuffer",
        });

        this.tetraDepthBuffer = this.device.createBuffer({
            size: this.numTetra * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.tetraDepthBuffer",
        });

        this.tetraUVBuffer = this.device.createBuffer({
            size: this.numTetra * 4 * 2 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.tetraDepthBuffer",
        });

        this.visibleFacesBuffer = this.device.createBuffer({
            size: this.numTetra * 4 * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.tetraDepthBuffer",
        });

        this.tileCountBuffer = this.device.createBuffer({
            size: this.numTetra * 4, // u32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.tileCountBuffer"
        });

        this.tileOffsetBuffer = this.device.createBuffer({
            size: this.scanPipeline.getAlignedSize(this.numTetra) * 4, // u32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.tileOffsetBuffer"
        });

        this.scanTileCounts = this.scanPipeline.prepareGPUInput(
            this.tileOffsetBuffer,
            this.scanPipeline.getAlignedSize(this.numTetra));

        // buffer for the range of tiles for each tetra
        this.rangesBuffer = this.device.createBuffer({
            size: Math.ceil(this.canvas.width / this.tileSize)  * Math.ceil(this.canvas.height / this.tileSize) * 4, // u32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.rangesBuffer"
        });

        // create a GPU buffer for the uniform data.
        this.uniformBuffer = this.device.createBuffer({
            size: uniformLayout.size,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.uniformBuffer",
        });

        // buffer for the num tetra, set once
        this.numTetraBuffer = this.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.numTetraBuffer"
        });
        this.device.queue.writeBuffer(
            this.numTetraBuffer,
            0,
            new Uint32Array([this.numTetra]),
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

        // buffer for the tile size, set once
        this.tileSizeBuffer = this.device.createBuffer({
            size: 1 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.tileSizeBuffer"
        });
        this.device.queue.writeBuffer(
            this.tileSizeBuffer,
            0,
            new Uint32Array([this.tileSize]),
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
            new Uint32Array([Math.ceil(this.canvas.width / this.tileSize) * Math.ceil(this.canvas.height / this.tileSize)]),
            0,
            1
        );

        this.processTetraPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: process_tetras,
                }),
                entryPoint: "main",
            },
        });
        this.processTetraBindGroup = this.device.createBindGroup({
            layout: this.processTetraPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.pointDataBuffer}},
                {binding: 1, resource: {buffer: this.tetraDataBuffer}},
                {binding: 2, resource: {buffer: this.tileCountBuffer}},
                {binding: 3, resource: {buffer: this.tetraRectBuffer}},
                {binding: 4, resource: {buffer: this.tetraDepthBuffer}},
                {binding: 5, resource: {buffer: this.tetraUVBuffer}},
                {binding: 6, resource: {buffer: this.visibleFacesBuffer}},
                {binding: 7, resource: {buffer: this.uniformBuffer}},
                {binding: 8, resource: {buffer: this.numTetraBuffer}},
                {binding: 9, resource: {buffer: this.canvasSizeBuffer}},
                {binding: 10, resource: {buffer: this.tileSizeBuffer}}
            ]
        });

        this.writeTileIDsPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: write_tile_ids,
                }),
                entryPoint: "main",
            },
        });

        this.renderTarget = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
                       GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST
        });
        this.renderTargetCopy = this.device.createTexture({
            size: [this.canvas.width, this.canvas.height, 1],
            format: "rgba8unorm",
            usage: GPUTextureUsage.STORAGE_BINDING | GPUTextureUsage.TEXTURE_BINDING |
                       GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST
        });
        this.computeTilesPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: compute_tiles.replace(/TILE_SIZE_MACRO/g, String(this.tileSize)),
                }),
                entryPoint: "main",
            },
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
        this.tetraDataBuffer.destroy();
        this.tetraDepthBuffer.destroy();
        this.tetraRectBuffer.destroy();
        this.numTetraBuffer.destroy();
        this.tetraIDBuffer.destroy();
        this.tileCountBuffer.destroy();
        this.tileOffsetBuffer.destroy();
        this.tileIDBuffer.destroy();
        this.rangesBuffer.destroy();
        this.canvasSizeBuffer.destroy();
        this.tileSizeBuffer.destroy();
        this.numTilesBuffer.destroy();
        this.renderTarget.destroy();
        this.renderTargetCopy.destroy();

        this.destroyCallback();
    }

    async animate() {
        if (this.destroyCallback !== null) {
            this.destroyImpl();
            return;
        }

        if (!this.interactiveCamera.isDirty()) {
            requestAnimationFrame(() => this.animate());
            return;
        }
        console.log(`++++++++ New frame ++++++++`);
        var totalStart = performance.now();

        const camera = this.interactiveCamera.getCamera();

        const position = camera.getPosition();

        const tanHalfFovX = 0.5 * this.canvas.width / camera.focalX;
        const tanHalfFovY = 0.5 * this.canvas.height / camera.focalY;

        this.depthSortMatrix = mat4toArrayOfArrays(camera.viewMatrix);

        let uniformsMatrixBuffer = new ArrayBuffer(this.uniformBuffer.size);
        let uniforms = {
            viewMatrix: mat4toArrayOfArrays(camera.viewMatrix),
            projMatrix: mat4toArrayOfArrays(camera.getProjMatrix()),
            cameraPosition: Array.from(position),
            tanHalfFovX: tanHalfFovX,
            tanHalfFovY: tanHalfFovY,
            focalX: camera.focalX,
            focalY: camera.focalY,
            scaleModifier: camera.scaleModifier,
        }
        
        console.log(uniforms);
        uniformLayout.pack(0, uniforms, new DataView(uniformsMatrixBuffer));

        this.device.queue.writeBuffer(
            this.uniformBuffer,
            0,
            uniformsMatrixBuffer,
            0,
            uniformsMatrixBuffer.byteLength
        );

        { 
            var start = performance.now();
            // compute the tile counts, depths, and bounding rect (in tile space) of each tetra
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.processTetraPipeline);
            passEncoder.setBindGroup(0, this.processTetraBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.numTetra / 256));
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            var end = performance.now();
            console.log(`Process tetra took ${end - start} ms`);
        }

        {
            var dbgBuffer = this.device.createBuffer({
                size: this.visibleFacesBuffer.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            var commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(this.visibleFacesBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await dbgBuffer.mapAsync(GPUMapMode.READ);

            var debugValsf = new Float32Array(dbgBuffer.getMappedRange());
            console.log(debugValsf);
        }
        // {
        //     var dbgBuffer = this.device.createBuffer({
        //         size: this.tetraRectBuffer.size,
        //         usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        //     });

        //     var commandEncoder = this.device.createCommandEncoder();
        //     commandEncoder.copyBufferToBuffer(this.tetraRectBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
        //     this.device.queue.submit([commandEncoder.finish()]);
        //     await this.device.queue.onSubmittedWorkDone();

        //     await dbgBuffer.mapAsync(GPUMapMode.READ);

        //     var debugValsf = new Uint32Array(dbgBuffer.getMappedRange());
        //     console.log(debugValsf);
        // }
        {
            var dbgBuffer = this.device.createBuffer({
                size: this.tetraUVBuffer.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            var commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(this.tetraUVBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await dbgBuffer.mapAsync(GPUMapMode.READ);

            var debugVals = new Float32Array(dbgBuffer.getMappedRange());
            console.log(debugVals);
            console.log(debugVals[2] - debugVals[0], debugVals[3] - debugVals[1])
            console.log(debugVals[4] - debugVals[0], debugVals[5] - debugVals[1])
            console.log((debugVals[2] - debugVals[0]) * (debugVals[5] - debugVals[1]) - (debugVals[4] - debugVals[0]) * (debugVals[3] - debugVals[1]))
        }

        // find the offsets for each tetra to write its tile intersections
        var commandEncoder = this.device.createCommandEncoder();
        // we scan the tileOffsetBuffer, so copy the tile count information over
        commandEncoder.copyBufferToBuffer(this.tileCountBuffer,
            0,
            this.tileOffsetBuffer,
            0,
            this.numTetra * 4);
        this.device.queue.submit([commandEncoder.finish()]);
        var start = performance.now();
        this.numIntersections = await this.scanTileCounts.scan(this.numTetra);
        var end = performance.now();
        console.log(`Scan tile counts took ${end - start} ms`);
        console.log(`Found ${this.numIntersections} intersections`);
        {
            var dbgBuffer = this.device.createBuffer({
                size: this.tileOffsetBuffer.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            var commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(this.tileOffsetBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await dbgBuffer.mapAsync(GPUMapMode.READ);

            var tileCountVals = new Uint32Array(dbgBuffer.getMappedRange());
            console.log(tileCountVals);
        }
        const sortBuffers = this.sorter.createSortBuffers(this.numIntersections);
        this.tileIDBuffer = sortBuffers.keys;
        this.tetraIDBuffer = sortBuffers.values;

        this.writeTileIDsBindGroup = this.device.createBindGroup({
            layout: this.writeTileIDsPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.tileOffsetBuffer}},
                {binding: 1, resource: {buffer: this.tetraRectBuffer}},
                {binding: 2, resource: {buffer: this.tetraDepthBuffer}},
                {binding: 3, resource: {buffer: this.tileIDBuffer}},
                {binding: 4, resource: {buffer: this.tetraIDBuffer}},
                {binding: 5, resource: {buffer: this.numTetraBuffer}},
                {binding: 6, resource: {buffer: this.canvasSizeBuffer}},
                {binding: 7, resource: {buffer: this.tileSizeBuffer}},
            ]
        });
        { 
            // write tile/depth combined IDs at computed offsets for each tetra
            var start = performance.now();
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.writeTileIDsPipeline);
            passEncoder.setBindGroup(0, this.writeTileIDsBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.numTetra / 256));
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            var end = performance.now();    
            console.log(`Write Tile IDs took ${end - start} ms`)
        }

        // sort tetra ids by the tile ids so each tile has all the tetras acting on it
        var start = performance.now();
        const sortEncoder = this.device.createCommandEncoder();
        this.sorter.sort(sortEncoder, this.device.queue, sortBuffers);
        this.device.queue.submit([sortEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        var end = performance.now();
        console.log(`Sort took ${end - start} ms`);

        // {
        //     var dbgBuffer = this.device.createBuffer({
        //         size: this.tetraIDBuffer.size,
        //         usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        //     });

        //     var commandEncoder = this.device.createCommandEncoder();
        //     commandEncoder.copyBufferToBuffer(this.tetraIDBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
        //     this.device.queue.submit([commandEncoder.finish()]);
        //     await this.device.queue.onSubmittedWorkDone();

        //     await dbgBuffer.mapAsync(GPUMapMode.READ);

        //     var tetraIDs = new Uint32Array(dbgBuffer.getMappedRange());
        //     console.log(tetraIDs);
        // }

        // {
        //     var dbgBuffer = this.device.createBuffer({
        //         size: this.tileIDBuffer.size,
        //         usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        //     });

        //     var commandEncoder = this.device.createCommandEncoder();
        //     commandEncoder.copyBufferToBuffer(this.tileIDBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
        //     this.device.queue.submit([commandEncoder.finish()]);
        //     await this.device.queue.onSubmittedWorkDone();

        //     await dbgBuffer.mapAsync(GPUMapMode.READ);

        //     var tileIDs = new Uint32Array(dbgBuffer.getMappedRange());
        //     console.log(tileIDs);
        // }

        { 
            // compute the ranges of IDs for each tile to work on
            var start = performance.now();
            this.computeRangesBindGroup = this.device.createBindGroup({
                layout: this.computeRangesPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.tileIDBuffer}},
                    {binding: 1, resource: {buffer: this.rangesBuffer}},
                    {binding: 2, resource: {buffer: this.numTilesBuffer}},
                ]
            });
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computeRangesPipeline);
            passEncoder.setBindGroup(0, this.computeRangesBindGroup);
            // workgroup is 256 and each thread does 64 elements
            passEncoder.dispatchWorkgroups(Math.ceil(this.numIntersections / (64 * 256)));
            passEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            var end = performance.now();
            console.log(`Compute tile ranges took ${end - start} ms`);
        }

        // // {
        // //     var dbgBuffer = this.device.createBuffer({
        // //         size: this.tileIDBuffer.size,
        // //         usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        // //     });

        // //     var commandEncoder = this.device.createCommandEncoder();
        // //     commandEncoder.copyBufferToBuffer(this.tileIDBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
        // //     this.device.queue.submit([commandEncoder.finish()]);
        // //     await this.device.queue.onSubmittedWorkDone();

        // //     await dbgBuffer.mapAsync(GPUMapMode.READ);

        // //     var debugValsf = new Uint32Array(dbgBuffer.getMappedRange());
        // //     console.log(debugValsf);
        // // }

        { 
            // compute the final image - each pixel accumulates the colors of the gaussians in its tile
            var start = performance.now();
            this.computeTilesBindGroup = this.device.createBindGroup({
                layout: this.computeTilesPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: this.renderTarget.createView()},
                    {binding: 1, resource: {buffer: this.rangesBuffer}},
                    {binding: 2, resource: {buffer: this.tetraIDBuffer}},
                    {binding: 3, resource: {buffer: this.tetraDataBuffer}},
                    {binding: 4, resource: {buffer: this.pointDataBuffer}},
                    {binding: 5, resource: {buffer: this.tetraUVBuffer}},
                    {binding: 6, resource: {buffer: this.canvasSizeBuffer}},
                    {binding: 7, resource: {buffer: this.tileSizeBuffer}},
                    {binding: 8, resource: {buffer: this.uniformBuffer}},
                ]
            });
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computeTilesPipeline);
            passEncoder.setBindGroup(0, this.computeTilesBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.canvas.width / this.tileSize), Math.ceil(this.canvas.height / this.tileSize));
            passEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            var end = performance.now();
            console.log(`Compute tiles took ${end - start} ms`);
        }

        { 
            // blit the computed image onto the screen
            var start = performance.now();
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
            await this.device.queue.onSubmittedWorkDone();
            var end = performance.now();
            console.log(`Rendering took ${end - start} ms`);
        }
        this.numFrames++;
        // clear everything for next pass
        var commandEncoder = this.device.createCommandEncoder();
        commandEncoder.clearBuffer(this.tileCountBuffer);
        commandEncoder.clearBuffer(this.tetraRectBuffer);
        commandEncoder.clearBuffer(this.tetraDepthBuffer);
        commandEncoder.clearBuffer(this.rangesBuffer);
        sortBuffers.destroy();
        commandEncoder.copyTextureToTexture(
            { texture: this.renderTargetCopy}, 
            { texture: this.renderTarget},
            { width: this.canvas.width, height: this.canvas.height, depthOrArrayLayers: 1 });
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        var totalEnd = performance.now();

        console.log(`TOTAL FRAME TIME: ${totalEnd - totalStart} ms`);
        console.log("------------------------------------------");
        requestAnimationFrame(() => this.animate());
    }
}