export function loadFileAsText(file: File): Promise<String> {
    /* loads a file as an ArrayBuffer (i.e. a binary blob) */
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (event) => {
            if (!event.target || !event.target.result) {
                reject('Failed to load file');
                return;
            }
            if (!(typeof event.target.result === "string")) {
                reject('Did not receive a text file');
                return;
            }
            resolve(event.target.result);
        };

        reader.onerror = (event) => {
            if (!event.target) {
                reject('Failed to load file');
                return;
            }
            reject(event.target.error);
        };

        reader.readAsText(file);
    });
}

export class Volume {
    /* A class that
        1) reads the text from a .off file
        2) converts internally into a structured representation
        3) packs the structured representation into a flat array of bytes as expected by the shaders
    */
    numPoints: number;
    numTetra: number;

    points: Float32Array;
    tetra: Uint32Array;

    constructor(text: string) {
        const lines = text.trim().split('\n');

        // parse the header
        [this.numPoints, this.numTetra] = lines[0].trim().split(' ').map(Number);
      
        // Parse the point data
        const pointData: number[] = [];
        for (let i = 1; i <= this.numPoints; i++) {
          const [x, y, z, value] = lines[i].trim().split(' ').map(Number);
          pointData.push(x, y, z, value);
        }
        this.points = new Float32Array(pointData);
      
        // Parse the tetrahedra data
        const tetrahedraData: number[] = [];
        for (let i = this.numPoints + 1; i <= this.numPoints + this.numTetra; i++) {
          const [idx0, idx1, idx2, idx3] = lines[i].trim().split(' ').map(Number);
          tetrahedraData.push(idx0, idx1, idx2, idx3);
        }
        this.tetra = new Uint32Array(tetrahedraData);    
    }
}