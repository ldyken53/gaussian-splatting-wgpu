// We use webpack to package our shaders as string resources that we can import
import shaderCode from "./triangle.wgsl";
import { RadixSorter } from "./radix_sort";
import { loadFileAsArrayBuffer, PackedGaussians } from "./ply";
import { Renderer } from "./renderer";

(async () => {
    if (navigator.gpu === undefined) {
        document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }

    // Get a GPU device to render with
    let adapter = await navigator.gpu.requestAdapter();
    let device = await adapter.requestDevice();

    // Grab needed HTML elements
    const plyFileInput = document.getElementById('plyButton') as HTMLInputElement;
    const loadingPopup = document.getElementById('loading-popup')! as HTMLDivElement;

    function handlePlyChange(event: any) {
        const file = event.target.files[0];
    
        async function onFileLoad(arrayBuffer: ArrayBuffer) {
            const gaussians = new PackedGaussians(arrayBuffer);
            console.log(gaussians);
            const renderer = new Renderer(canvas, device, gaussians);
            loadingPopup.style.display = 'none'; // hide loading popup
        }
    
        if (file) {
            loadingPopup.style.display = 'block'; // show loading popup
            loadFileAsArrayBuffer(file)
                .then(onFileLoad);
        }
    }
    plyFileInput!.addEventListener('change', handlePlyChange);

    let radixSorter = new RadixSorter(device);
    let keys = [8.4, 6.5, 0, 6.0, 6.77, -1, -2, 10, 9, 4];
    let keyBuf = device.createBuffer({
        mappedAtCreation: true,
        size: radixSorter.getAlignedSize(10) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    new Float32Array(keyBuf.getMappedRange()).set(new Float32Array(keys));
    keyBuf.unmap();
    let values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let valueBuf = device.createBuffer({
        mappedAtCreation: true,
        size: radixSorter.getAlignedSize(10) * 4,
        usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    new Uint32Array(valueBuf.getMappedRange()).set(new Uint32Array(values));
    valueBuf.unmap();
    await radixSorter.sort(keyBuf, valueBuf, 10, false, true);
    let debugBuf = device.createBuffer({
        size: radixSorter.getAlignedSize(10) * 4,
        usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    });
    var commandEncoder = device.createCommandEncoder();
    commandEncoder.copyBufferToBuffer(valueBuf, 0, debugBuf, 0, debugBuf.size);
    device.queue.submit([commandEncoder.finish()]);
    await device.queue.onSubmittedWorkDone(); 
    await debugBuf.mapAsync(GPUMapMode.READ);
    var debugVals = new Uint32Array(debugBuf.getMappedRange());
    console.log(debugVals);

    // Get a context to display our rendered image on the canvas
    let canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
    let context = canvas.getContext("webgpu");

    // Setup shader modules
    let shaderModule = device.createShaderModule({code: shaderCode});
    let compilationInfo = await shaderModule.getCompilationInfo();
    if (compilationInfo.messages.length > 0) {
        let hadError = false;
        console.log("Shader compilation log:");
        for (let i = 0; i < compilationInfo.messages.length; ++i) {
            let msg = compilationInfo.messages[i];
            console.log(`${msg.lineNum}:${msg.linePos} - ${msg.message}`);
            hadError = hadError || msg.type == "error";
        }
        if (hadError) {
            console.log("Shader failed to compile");
            return;
        }
    }

    // Specify vertex data
    // Allocate room for the vertex data: 3 vertices, each with 2 float4's
    let dataBuf = device.createBuffer(
        {size: 3 * 2 * 4 * 4, usage: GPUBufferUsage.VERTEX, mappedAtCreation: true});

    // Interleaved positions and colors
    new Float32Array(dataBuf.getMappedRange()).set([
        1, -1, 0, 1,  // position
        1, 0, 0, 1,  // color
        -1, -1, 0, 1,  // position
        0, 1, 0, 1,  // color
        0, 1, 0, 1,  // position
        0, 0, 1, 1,  // color
    ]);
    dataBuf.unmap();

    // Vertex attribute state and shader stage
    let vertexState = {
        // Shader stage info
        module: shaderModule,
        entryPoint: "vertex_main",
        // Vertex buffer info
        buffers: [{
            arrayStride: 2 * 4 * 4,
            attributes: [
                {format: "float32x4" as GPUVertexFormat, offset: 0, shaderLocation: 0},
                {format: "float32x4" as GPUVertexFormat, offset: 4 * 4, shaderLocation: 1}
            ]
        }]
    };

    // Setup render outputs
    let swapChainFormat = "bgra8unorm" as GPUTextureFormat;
    context.configure(
        {device: device, format: swapChainFormat, usage: GPUTextureUsage.RENDER_ATTACHMENT});

    let depthFormat = "depth24plus-stencil8" as GPUTextureFormat;
    let depthTexture = device.createTexture({
        size: {width: canvas.width, height: canvas.height, depthOrArrayLayers: 1},
        format: depthFormat,
        usage: GPUTextureUsage.RENDER_ATTACHMENT
    });

    let fragmentState = {
        // Shader info
        module: shaderModule,
        entryPoint: "fragment_main",
        // Output render target info
        targets: [{format: swapChainFormat}]
    };

    // Create render pipeline
    let layout = device.createPipelineLayout({bindGroupLayouts: []});

    let renderPipeline = device.createRenderPipeline({
        layout: layout,
        vertex: vertexState,
        fragment: fragmentState,
        depthStencil: {format: depthFormat, depthWriteEnabled: true, depthCompare: "less"}
    });

    let renderPassDesc = {
        colorAttachments: [{
            view: null as GPUTextureView,
            loadOp: "clear" as GPULoadOp,
            clearValue: [0.3, 0.3, 0.3, 1],
            storeOp: "store" as GPUStoreOp
        }],
        depthStencilAttachment: {
            view: depthTexture.createView(),
            depthLoadOp: "clear" as GPULoadOp,
            depthClearValue: 1.0,
            depthStoreOp: "store" as GPUStoreOp,
            stencilLoadOp: "clear" as GPULoadOp,
            stencilClearValue: 0,
            stencilStoreOp: "store" as GPUStoreOp
        }
    };

    // Render!
    const render = async () => {
        renderPassDesc.colorAttachments[0].view = context.getCurrentTexture().createView();

        let commandEncoder = device.createCommandEncoder();

        let renderPass = commandEncoder.beginRenderPass(renderPassDesc);

        renderPass.setPipeline(renderPipeline);
        renderPass.setVertexBuffer(0, dataBuf);
        renderPass.draw(3, 1, 0, 0);

        renderPass.end();
        device.queue.submit([commandEncoder.finish()]);
        requestAnimationFrame(render);
    }
    requestAnimationFrame(render);
})();
