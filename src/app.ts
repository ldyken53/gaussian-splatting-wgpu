import { CameraFileParser, InteractiveCamera } from "./camera";
import { loadFileAsArrayBuffer, PackedGaussians } from "./ply";
import { Renderer } from "./renderer";
import { OpacityTransferFunction } from "./OpacityTransferFunction";
import { guessWorkgroupSize } from "./radix_sort/utils";

(async () => {
    if (navigator.gpu === undefined) {
        document.getElementById("webgpu-canvas").setAttribute("style", "display:none;");
        document.getElementById("no-webgpu").setAttribute("style", "display:block;");
        return;
    }

    // Get a GPU device to render with
    let adapter = await navigator.gpu.requestAdapter();
    console.log(adapter.limits);
    let device = await adapter.requestDevice({
        requiredLimits: {
            maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
            maxBufferSize: adapter.limits.maxBufferSize,
            maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
            maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize
        }
    });
    
    // Grab needed HTML elements
    const plyFileInput = document.getElementById('plyButton') as HTMLInputElement;
    const loadingPopup = document.getElementById('loading-popup')! as HTMLDivElement;
    const tileSizeSlider = document.getElementById('tileSize') as HTMLInputElement;
    tileSizeSlider.value = "16";
    const weightSlider = document.getElementById('weightSlider') as HTMLInputElement;
    let canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;
    const cameraFileInput = document.getElementById('cameraButton')! as HTMLInputElement;
    const cameraList = document.getElementById('cameraList')! as HTMLUListElement;
    let interactiveCamera = InteractiveCamera.default(canvas);
    var currentRenderer: Renderer;
    let opacityTransferFunction = new OpacityTransferFunction();

    function handlePlyChange(event: any) {
        const file = event.target.files[0];
    
        async function onFileLoad(arrayBuffer: ArrayBuffer) {
            if (currentRenderer) {
                await currentRenderer.destroy();
            }

            const gaussians = new PackedGaussians(arrayBuffer);
            const renderer = new Renderer(canvas, interactiveCamera, device, gaussians, parseInt(tileSizeSlider.value));
            weightSlider.oninput = () => {
                renderer.updateWeight(parseFloat(weightSlider!.value));
                document.getElementById('weightValue')!.innerText=weightSlider!.value;
            };            
            opacityTransferFunction.setUpdateCallback((buffer: number[]) => {
                renderer.updateOpacity(buffer);
            });
            currentRenderer = renderer;
            loadingPopup.style.display = 'none'; // hide loading popup
        }
    
        if (file) {
            loadingPopup.style.display = 'block'; // show loading popup
            loadFileAsArrayBuffer(file)
                .then(onFileLoad);
        }
    }
    plyFileInput!.addEventListener('change', handlePlyChange);
    new CameraFileParser(
        cameraFileInput,
        cameraList,
        canvas,
        (camera) => interactiveCamera.setNewCamera(camera),
    );

})();
