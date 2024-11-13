import { Mat4 } from 'wgpu-matrix';
import { saveAs } from 'file-saver';

import { PackedGaussians } from './ply';
import { Struct, f32, mat4x4, vec3 } from './packing';
import { ExclusiveScanPipeline, ExclusiveScanner } from './exclusive_scan';
import { InteractiveCamera } from './camera';

import compute_aabbs from "./compute_aabbs.wgsl";
import compute_cells from "./compute_cells.wgsl";
import compute_ranges from "./compute_ranges.wgsl";
import write_cell_ids from "./write_cell_ids.wgsl";
import { GPUSorter } from './radix_sort/sort';

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

    numGaussians: number;
    numIntersections: number;
    numFrames: number;

    device: GPUDevice;
    contextGpu: GPUCanvasContext;

    sorter: GPUSorter;
    scanPipeline: ExclusiveScanPipeline;
    scanCellCounts: ExclusiveScanner;

    uniformBuffer: GPUBuffer; // camera uniforms
    pointDataBuffer: GPUBuffer;
    gaussianDataBuffer: GPUBuffer;
    gaussianIDBuffer: GPUBuffer; // buffer of gaussian indices (used for sort by cell)
    cellCountBuffer: GPUBuffer; // used to count the number of cell intersections for each Gaussian
    cellOffsetBuffer: GPUBuffer; // filled with output of prefix sum on cellCountBuffer
    cellIDBuffer: GPUBuffer; // cell IDs for each gaussian
    rangesBuffer: GPUBuffer; // intersection ranges for each cell

    numGaussianBuffer: GPUBuffer;

    renderTarget: GPUTexture;
    renderTargetCopy: GPUTexture;

    renderPipelineBindGroup: GPUBindGroup;
    pointDataBindGroup: GPUBindGroup;
    computeAABBsBindGroup: GPUBindGroup;
    writeCellIDsBindGroup: GPUBindGroup;
    computeRangesBindGroup: GPUBindGroup;

    renderPipeline: GPURenderPipeline;
    computeAABBsPipeline: GPUComputePipeline;
    writeCellIDsPipeline: GPUComputePipeline;
    computeRangesPipeline: GPUComputePipeline;

    depthSortMatrix: number[][];

    // fps counter
    fpsCounter: HTMLLabelElement;
    lastDraw: number;

    destroyCallback: (() => void) | null = null;
    numIntersectionsBuffer: GPUBuffer;
    volumeMins: number[];
    volumeMaxes: number[];
    cellSize: number;
    volumeInfoBuffer: GPUBuffer;
    numCells: number[];
    computeCellsPipeline: GPUComputePipeline;
    computeCellsBindGroup: GPUBindGroup;
    cellDataBuffer: GPUBuffer;

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
        gaussians: PackedGaussians,
        tileSize: number
    ) {
        this.volumeMins = gaussians.mins; 
        this.volumeMaxes = gaussians.maxes;
        // this.volumeMins = [-0.75, -0.3, 2.8]; 
        // this.volumeMaxes = [0.75, 0.3, 3.2];
        this.cellSize = 0.02;
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

        this.numGaussians = gaussians.numGaussians;
        console.log(`Num Gaussians: ${this.numGaussians}`);

        this.pointDataBuffer = this.device.createBuffer({
            size: gaussians.gaussianArrayLayout.size,
            usage: GPUBufferUsage.STORAGE,
            mappedAtCreation: true,
            label: "renderer.pointDataBuffer",
        });
        new Uint8Array(this.pointDataBuffer.getMappedRange()).set(new Uint8Array(gaussians.gaussiansBuffer));
        this.pointDataBuffer.unmap();

        this.cellCountBuffer = this.device.createBuffer({
            size: this.numGaussians * 4, // u32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.cellCountBuffer"
        });

        this.cellOffsetBuffer = this.device.createBuffer({
            size: this.scanPipeline.getAlignedSize(this.numGaussians) * 4, // u32
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.cellOffsetBuffer"
        });

        this.scanCellCounts = this.scanPipeline.prepareGPUInput(
            this.cellOffsetBuffer,
            this.scanPipeline.getAlignedSize(this.numGaussians));

        // buffer for gaussian info needed for computing cells
        this.gaussianDataBuffer = this.device.createBuffer({
            size: this.numGaussians * (24) * 4, // vec3, vec3, f32 with alignment rules
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.gaussianDataBuffer"
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

        this.volumeInfoBuffer = this.device.createBuffer({
            size: 8 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.volumeInfoBuffer"
        });
        this.device.queue.writeBuffer(
            this.volumeInfoBuffer,
            0,
            new Float32Array([this.volumeMins[0], this.volumeMins[1], this.volumeMins[2], this.cellSize, this.volumeMaxes[0], this.volumeMaxes[1], this.volumeMaxes[2]]),
            0,
            7
        );
        this.numCells = [
            Math.ceil((this.volumeMaxes[0] - this.volumeMins[0]) / this.cellSize),
            Math.ceil((this.volumeMaxes[1] - this.volumeMins[1]) / this.cellSize),
            Math.ceil((this.volumeMaxes[2] - this.volumeMins[2]) / this.cellSize)
        ];
        console.log(`Cells per dimension:
            x: ${this.numCells[0]}
            y: ${this.numCells[1]}
            z: ${this.numCells[2]}
        `);

        // buffer for output data for each cell
        this.cellDataBuffer = this.device.createBuffer({
            size: this.numCells[0] * this.numCells[1] * this.numCells[2] * 4, // f32 for each cell
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.cellDataBuffer"
        });

        // buffer for the range of gaussians for each cell
        this.rangesBuffer = this.device.createBuffer({
            size: this.numCells[0] * this.numCells[1] * this.numCells[2] * 2 * 4, // vec2<u32>
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            label: "renderer.rangesBuffer"
        });

        this.computeAABBsPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: compute_aabbs,
                }),
                entryPoint: "main",
            },
        });
        this.computeAABBsBindGroup = this.device.createBindGroup({
            layout: this.computeAABBsPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.pointDataBuffer}},
                {binding: 1, resource: {buffer: this.gaussianDataBuffer}},
                {binding: 2, resource: {buffer: this.cellCountBuffer}},
                {binding: 3, resource: {buffer: this.numGaussianBuffer}},
                {binding: 4, resource: {buffer: this.volumeInfoBuffer}},
            ]
        });

        this.writeCellIDsPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: write_cell_ids,
                }),
                entryPoint: "main",
            },
        });

        this.computeCellsPipeline = this.device.createComputePipeline({
            layout: "auto",
            compute: {
                module: this.device.createShaderModule({
                    code: compute_cells,
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

        // start the animation loop
        requestAnimationFrame(() => this.animate());
    }

    private destroyImpl(): void {
        if (this.destroyCallback === null) {
            throw new Error("destroyImpl called without destroyCallback set!");
        }

        this.uniformBuffer.destroy();
        this.pointDataBuffer.destroy();
        this.gaussianDataBuffer.destroy();
        this.gaussianIDBuffer.destroy();
        this.cellCountBuffer.destroy();
        this.cellOffsetBuffer.destroy();
        this.cellIDBuffer.destroy();
        this.rangesBuffer.destroy();
        this.numGaussianBuffer.destroy();
        this.volumeInfoBuffer.destroy();
        this.cellDataBuffer.destroy();

        this.destroyCallback();
    }

    async animate() {
        if (this.destroyCallback !== null) {
            this.destroyImpl();
            return;
        }

        console.log(`++++++++ New frame ++++++++`);
        var totalStart = performance.now();

        { 
            var start = performance.now();
            // compute the cell counts
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computeAABBsPipeline);
            passEncoder.setBindGroup(0, this.computeAABBsBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.numGaussians / 256));
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            var end = performance.now();
            console.log(`Compute aabbs took ${end - start} ms`);
        }

        {
            var dbgBuffer = this.device.createBuffer({
                size: this.cellCountBuffer.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            var commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(this.cellCountBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await dbgBuffer.mapAsync(GPUMapMode.READ);

            var cellCountVals = new Float32Array(dbgBuffer.getMappedRange());
            console.log(cellCountVals);
            let count = 0;
            for (let i of cellCountVals) {
                if (i == 0) {
                    count += 1;
                }
            }
            console.log(count);
        }

        // // find the offsets for each gaussian to write its cell intersections
        var commandEncoder = this.device.createCommandEncoder();
        // we scan the cellOffsetBuffer, so copy the cell count information over
        commandEncoder.copyBufferToBuffer(this.cellCountBuffer,
            0,
            this.cellOffsetBuffer,
            0,
            this.numGaussians * 4);
        this.device.queue.submit([commandEncoder.finish()]);
        var start = performance.now();
        this.numIntersections = await this.scanCellCounts.scan(this.numGaussians);
        var end = performance.now();
        console.log(`Scan cell counts took ${end - start} ms`);
        console.log(`Found ${this.numIntersections} intersections`);
        this.numIntersectionsBuffer = this.device.createBuffer({
            size: 1 * 4,
            usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            label: "renderer.numIntersectionsBuffer"
        });
        this.device.queue.writeBuffer(
            this.numIntersectionsBuffer,
            0,
            new Uint32Array([this.numIntersections]),
            0,
            1
        );
        // {
        //     var dbgBuffer = this.device.createBuffer({
        //         size: this.cellOffsetBuffer.size,
        //         usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        //     });

        //     var commandEncoder = this.device.createCommandEncoder();
        //     commandEncoder.copyBufferToBuffer(this.cellOffsetBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
        //     this.device.queue.submit([commandEncoder.finish()]);
        //     await this.device.queue.onSubmittedWorkDone();

        //     await dbgBuffer.mapAsync(GPUMapMode.READ);

        //     var cellCountVals = new Uint32Array(dbgBuffer.getMappedRange());
        //     console.log(cellCountVals);
        // }
        const sortBuffers = this.sorter.createSortBuffers(this.numIntersections);
        this.cellIDBuffer = sortBuffers.keys;
        this.gaussianIDBuffer = sortBuffers.values;

        this.writeCellIDsBindGroup = this.device.createBindGroup({
            layout: this.writeCellIDsPipeline.getBindGroupLayout(0),
            entries: [
                {binding: 0, resource: {buffer: this.cellOffsetBuffer}},
                {binding: 1, resource: {buffer: this.gaussianDataBuffer}},
                {binding: 2, resource: {buffer: this.cellIDBuffer}},
                {binding: 3, resource: {buffer: this.gaussianIDBuffer}},
                {binding: 4, resource: {buffer: this.numGaussianBuffer}},
                {binding: 5, resource: {buffer: this.volumeInfoBuffer}},
            ]
        });
        { 
            // write cell IDs at computed offsets for each gaussian
            var start = performance.now();
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.writeCellIDsPipeline);
            passEncoder.setBindGroup(0, this.writeCellIDsBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.numGaussians / 256));
            passEncoder.end();
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            var end = performance.now();    
            console.log(`Write cell IDs took ${end - start} ms`)
        }

        // // sort gaussian ids by the cell ids so each cell has all the gaussians acting on it
        var start = performance.now();
        const sortEncoder = this.device.createCommandEncoder();
        this.sorter.sort(sortEncoder, this.device.queue, sortBuffers);
        this.device.queue.submit([sortEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        var end = performance.now();
        console.log(`Sort took ${end - start} ms`);

        { 
            // compute the ranges of IDs for each cell to work on
            var start = performance.now();
            this.computeRangesBindGroup = this.device.createBindGroup({
                layout: this.computeRangesPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.cellIDBuffer}},
                    {binding: 1, resource: {buffer: this.rangesBuffer}},
                    {binding: 2, resource: {buffer: this.numIntersectionsBuffer}},
                ]
            });
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computeRangesPipeline);
            passEncoder.setBindGroup(0, this.computeRangesBindGroup);
            // workgroup is 256 and each thread does 64 elements
            passEncoder.dispatchWorkgroups(Math.ceil(this.numIntersections / 256));
            passEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            var end = performance.now();
            console.log(`Compute cell ranges took ${end - start} ms`);
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

            var debugValsf = new Uint32Array(dbgBuffer.getMappedRange());
            console.log(debugValsf);
        }

        { 
            // compute the final image - each cell averages the values of the gaussians in it
            var start = performance.now();
            this.computeCellsBindGroup = this.device.createBindGroup({
                layout: this.computeCellsPipeline.getBindGroupLayout(0),
                entries: [
                    {binding: 0, resource: {buffer: this.cellDataBuffer}},
                    {binding: 1, resource: {buffer: this.rangesBuffer}},
                    {binding: 2, resource: {buffer: this.gaussianIDBuffer}},
                    {binding: 3, resource: {buffer: this.gaussianDataBuffer}},
                    {binding: 4, resource: {buffer: this.volumeInfoBuffer}},
                ]
            });
            const commandEncoder = this.device.createCommandEncoder();
            const passEncoder = commandEncoder.beginComputePass();
            passEncoder.setPipeline(this.computeCellsPipeline);
            passEncoder.setBindGroup(0, this.computeCellsBindGroup);
            passEncoder.dispatchWorkgroups(Math.ceil(this.numCells[0] * this.numCells[1] * this.numCells[2] / 256));
            passEncoder.end();

            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();
            var end = performance.now();
            console.log(`Compute cells took ${end - start} ms`);
        }

        {
            var dbgBuffer = this.device.createBuffer({
                size: this.cellDataBuffer.size,
                usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
            });

            var commandEncoder = this.device.createCommandEncoder();
            commandEncoder.copyBufferToBuffer(this.cellDataBuffer, 0, dbgBuffer, 0, dbgBuffer.size);
            this.device.queue.submit([commandEncoder.finish()]);
            await this.device.queue.onSubmittedWorkDone();

            await dbgBuffer.mapAsync(GPUMapMode.READ);

            var debugVals = new Float32Array(dbgBuffer.getMappedRange());
            console.log(debugVals);
        }

        this.numFrames++;
        // clear everything for next pass
        var commandEncoder = this.device.createCommandEncoder();
        commandEncoder.clearBuffer(this.cellCountBuffer);
        commandEncoder.clearBuffer(this.gaussianDataBuffer);
        commandEncoder.clearBuffer(this.rangesBuffer);
        commandEncoder.clearBuffer(this.cellDataBuffer);
        sortBuffers.destroy();
        this.device.queue.submit([commandEncoder.finish()]);
        await this.device.queue.onSubmittedWorkDone();
        var totalEnd = performance.now();

        console.log(`TOTAL FRAME TIME: ${totalEnd - totalStart} ms`);
        console.log("------------------------------------------");
        // requestAnimationFrame(() => this.animate());

        let vtkContent = '';

        // Header information
        vtkContent += '# vtk DataFile Version 3.0\n';
        vtkContent += 'Volume Data Example\n';
        vtkContent += 'ASCII\n';
        vtkContent += 'DATASET STRUCTURED_POINTS\n';

        // Grid dimensions
        vtkContent += `DIMENSIONS ${this.numCells[0]} ${this.numCells[1]} ${this.numCells[2]}\n`;
        vtkContent += 'SPACING 1.0 1.0 1.0\n';
        vtkContent += 'ORIGIN 0.0 0.0 0.0\n';

        // Scalar field data
        vtkContent += `POINT_DATA ${this.numCells[0] * this.numCells[1] * this.numCells[2]}\n`;
        vtkContent += 'SCALARS volume_scalars float 1\n';
        vtkContent += 'LOOKUP_TABLE default\n';

        // Add scalar values in row-major order
        for (let i = 0; i < debugVals.length; i++) {
            vtkContent += debugVals[i].toFixed(6) + '\n';  // 6 decimal places for precision
        }

        const blob = new Blob([vtkContent], { type: 'text/plain;charset=utf-8' });
        saveAs(blob, "volume_cells.vtk");
    }
}