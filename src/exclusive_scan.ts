import prefix_sum from "./prefix_sum.wgsl";
import block_prefix_sum from "./block_prefix_sum.wgsl";
import add_block_sums from "./add_block_sums.wgsl";

var ScanBlockSize = 512;
export class ExclusiveScanPipeline {
    device: GPUDevice;
    workGroupSize: number;
    maxScanSize: number;
    clearBuf: GPUBuffer;
    scanBlocksLayout: GPUBindGroupLayout;
    scanBlockResultsLayout: GPUBindGroupLayout;
    scanBlocksPipeline: GPUComputePipeline;
    scanBlockResultsPipeline: GPUComputePipeline;
    addBlockSumsPipeline: GPUComputePipeline;
    getAlignedSize: (size: number) => number;
    // prepareInput: (cpuArray: any) => any;
    prepareGPUInput: (gpuBuffer: GPUBuffer, alignedSize: number) => ExclusiveScanner;
    constructor(device: GPUDevice) {
        this.device = device;
        // Each thread in a work group is responsible for 2 elements
        this.workGroupSize = ScanBlockSize / 2;
        // The max size which can be scanned by a single batch without carry in/out
        this.maxScanSize = ScanBlockSize * ScanBlockSize;
        console.log(`Block size: ${ScanBlockSize}, max scan size: ${this.maxScanSize}`);
    
        // Buffer to clear the block sums for each new scan
        var clearBlocks = device.createBuffer({
            size: ScanBlockSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            mappedAtCreation: true,
        });
        new Uint32Array(clearBlocks.getMappedRange()).fill(0);
        clearBlocks.unmap();
        this.clearBuf = clearBlocks;
    
        this.scanBlocksLayout = device.createBindGroupLayout({
            entries: [
                {binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: {type: "storage"}},
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                },
            ],
        });
    
        this.scanBlockResultsLayout = device.createBindGroupLayout({
            entries: [
                {
                    binding: 0,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                },
                {
                    binding: 1,
                    visibility: GPUShaderStage.COMPUTE,
                    buffer: {
                        type: "storage",
                    }
                },
            ],
        });
    
        this.scanBlocksPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.scanBlocksLayout],
            }),
            compute: {
                module: device.createShaderModule({code: prefix_sum}),
                entryPoint: "main",
            },
        });
    
        this.scanBlockResultsPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.scanBlockResultsLayout],
            }),
            compute: {
                module: device.createShaderModule({code: block_prefix_sum}),
                entryPoint: "main",
            },
        });
    
        this.addBlockSumsPipeline = device.createComputePipeline({
            layout: device.createPipelineLayout({
                bindGroupLayouts: [this.scanBlocksLayout],
            }),
            compute: {
                module: device.createShaderModule({code: add_block_sums}),
                entryPoint: "main",
            },
        });
    }

}
export var alignTo = function(val: number, align: number) {
    return Math.floor((val + align - 1) / align) * align;
};

// Serial scan for validation
// var serialExclusiveScan = function(array, output) {
//     output[0] = 0;
//     for (var i = 1; i < array.length; ++i) {
//         output[i] = array[i - 1] + output[i - 1];
//     }
//     return output[array.length - 1] + array[array.length - 1];
// };

ExclusiveScanPipeline.prototype.getAlignedSize = function(size: number) {
    return alignTo(size, ScanBlockSize);
};

// TODO: refactor to have this return a prepared scanner object?
// Then the pipelines and bind group layouts can be re-used and shared between the scanners
// ExclusiveScanPipeline.prototype.prepareInput = function(cpuArray) {
//     var alignedSize = alignTo(cpuArray.length, ScanBlockSize);

//     // Upload input and pad to block size elements
//     var inputBuf = this.device.createBuffer({
//         size: alignedSize * 4,
//         usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
//         mappedAtCreation: true,
//     });
//     new Uint32Array(inputBuf.getMappedRange()).set(cpuArray);
//     inputBuf.unmap();

//     return new ExclusiveScanner(this, inputBuf, alignedSize, cpuArray.length);
// };

ExclusiveScanPipeline.prototype.prepareGPUInput = function(gpuBuffer, alignedSize) {
    if (this.getAlignedSize(alignedSize) != alignedSize) {
        alert("Error: GPU input must be aligned to getAlignedSize");
    }

    return new ExclusiveScanner(this, gpuBuffer, alignedSize);
};

export class ExclusiveScanner {
    scanPipeline: ExclusiveScanPipeline;
    inputSize: number;
    inputBuf: GPUBuffer;
    readbackBuf: GPUBuffer;
    blockSumBuf: GPUBuffer;
    carryBuf: GPUBuffer;
    carryIntermediateBuf: GPUBuffer;
    scanBlockResultsBindGroup: GPUBindGroup;
    scan: (dataSize: number) => Promise<number>;
    constructor(scanPipeline: ExclusiveScanPipeline, gpuBuffer: GPUBuffer, alignedSize: number) {
        this.scanPipeline = scanPipeline;
        this.inputSize = alignedSize;
        this.inputBuf = gpuBuffer;

        this.readbackBuf = scanPipeline.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
        });

        // Block sum buffer
        var blockSumBuf = scanPipeline.device.createBuffer({
            size: ScanBlockSize * 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(blockSumBuf.getMappedRange()).fill(0);
        blockSumBuf.unmap();
        this.blockSumBuf = blockSumBuf;

        var carryBuf = scanPipeline.device.createBuffer({
            size: 8,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
            mappedAtCreation: true,
        });
        new Uint32Array(carryBuf.getMappedRange()).fill(0);
        carryBuf.unmap();
        this.carryBuf = carryBuf;

        // Can't copy from a buffer to itself so we need an intermediate to move the carry
        this.carryIntermediateBuf = scanPipeline.device.createBuffer({
            size: 4,
            usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
        });

        this.scanBlockResultsBindGroup = scanPipeline.device.createBindGroup({
            layout: this.scanPipeline.scanBlockResultsLayout,
            entries: [
                {
                    binding: 0,
                    resource: {
                        buffer: blockSumBuf,
                    },
                },
                {
                    binding: 1,
                    resource: {
                        buffer: carryBuf,
                    },
                },
            ],
        });
    }
}

ExclusiveScanner.prototype.scan = async function(dataSize) {
    // If the data size we're scanning within the larger input array has changed,
    // we just need to re-record the scan commands
    var numChunks = Math.ceil(dataSize / this.scanPipeline.maxScanSize);
    this.offsets = new Uint32Array(numChunks);
    for (var i = 0; i < numChunks; ++i) {
        this.offsets.set([i * this.scanPipeline.maxScanSize * 4], i);
    }

    // Scan through the data in chunks, updating carry in/out at the end to carry
    // over the results of the previous chunks
    var commandEncoder = this.scanPipeline.device.createCommandEncoder();

    // Clear the carry buffer and the readback sum entry if it's not scan size aligned
    commandEncoder.copyBufferToBuffer(this.scanPipeline.clearBuf, 0, this.carryBuf, 0, 8);
    for (var i = 0; i < numChunks; ++i) {
        var nWorkGroups =
            Math.min((this.inputSize - i * this.scanPipeline.maxScanSize) / ScanBlockSize,
                     ScanBlockSize);

        var scanBlockBG = null;
        if (nWorkGroups === ScanBlockSize) {
            scanBlockBG = this.scanPipeline.device.createBindGroup({
                layout: this.scanPipeline.scanBlocksLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.inputBuf,
                            size: Math.min(this.scanPipeline.maxScanSize, this.inputSize) * 4,
                            offset: this.offsets[i],
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.blockSumBuf,
                        },
                    },
                ],
            });
        } else {
            // Bind groups for processing the remainder if the aligned size isn't
            // an even multiple of the max scan size
            scanBlockBG = this.scanPipeline.device.createBindGroup({
                layout: this.scanPipeline.scanBlocksLayout,
                entries: [
                    {
                        binding: 0,
                        resource: {
                            buffer: this.inputBuf,
                            size: (this.inputSize % this.scanPipeline.maxScanSize) * 4,
                            offset: this.offsets[i],
                        },
                    },
                    {
                        binding: 1,
                        resource: {
                            buffer: this.blockSumBuf,
                        },
                    },
                ],
            });
        }

        // Clear the previous block sums
        commandEncoder.copyBufferToBuffer(
            this.scanPipeline.clearBuf, 0, this.blockSumBuf, 0, ScanBlockSize * 4);

        var computePass = commandEncoder.beginComputePass();

        computePass.setPipeline(this.scanPipeline.scanBlocksPipeline);
        computePass.setBindGroup(0, scanBlockBG);
        computePass.dispatchWorkgroups(nWorkGroups, 1, 1);

        computePass.setPipeline(this.scanPipeline.scanBlockResultsPipeline);
        computePass.setBindGroup(0, this.scanBlockResultsBindGroup);
        computePass.dispatchWorkgroups(1, 1, 1);

        computePass.setPipeline(this.scanPipeline.addBlockSumsPipeline);
        computePass.setBindGroup(0, scanBlockBG);
        computePass.dispatchWorkgroups(nWorkGroups, 1, 1);

        computePass.end();

        // Update the carry in value for the next chunk, copy carry out to carry in
        commandEncoder.copyBufferToBuffer(this.carryBuf, 4, this.carryIntermediateBuf, 0, 4);
        commandEncoder.copyBufferToBuffer(this.carryIntermediateBuf, 0, this.carryBuf, 0, 4);
    }
    var commandBuffer = commandEncoder.finish();

    // We need to clear a different element in the input buf for the last item if the data size
    // shrinks
    if (dataSize < this.inputSize) {
        var commandEncoder = this.scanPipeline.device.createCommandEncoder();
        commandEncoder.copyBufferToBuffer(
            this.scanPipeline.clearBuf, 0, this.inputBuf, dataSize * 4, 4);
        this.scanPipeline.device.queue.submit([commandEncoder.finish()]);
    }

    this.scanPipeline.device.queue.submit([commandBuffer]);

    // Readback the the last element to return the total sum as well
    var commandEncoder = this.scanPipeline.device.createCommandEncoder();
    if (dataSize < this.inputSize) {
        commandEncoder.copyBufferToBuffer(this.inputBuf, dataSize * 4, this.readbackBuf, 0, 4);
    } else {
        commandEncoder.copyBufferToBuffer(this.carryBuf, 4, this.readbackBuf, 0, 4);
    }
    this.scanPipeline.device.queue.submit([commandEncoder.finish()]);

    await this.readbackBuf.mapAsync(GPUMapMode.READ);
    var mapping = new Uint32Array(this.readbackBuf.getMappedRange());
    var sum = mapping[0];
    this.readbackBuf.unmap();

    return sum;
};
