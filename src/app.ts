// We use webpack to package our shaders as string resources that we can import
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
    let canvas = document.getElementById("webgpu-canvas") as HTMLCanvasElement;

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

})();
