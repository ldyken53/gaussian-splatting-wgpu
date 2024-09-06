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
        var mins = [10.0, 10.0, 10.0, 10.0];
        var maxes = [-10.0, -10.0, -10.0, -10.0];
        for (let i = 1; i <= this.numPoints; i++) {
          const [x, y, z, value] = lines[i].trim().split(' ').map(Number);
          pointData.push(x, y, z, value);
          if (x > maxes[0]) { maxes[0] = x }
          if (x < mins[0]) { mins[0] = x }
          if (y > maxes[1]) { maxes[1] = y }
          if (y < mins[1]) { mins[1] = y }
          if (z > maxes[2]) { maxes[2] = z }
          if (z < mins[2]) { mins[2] = z }
          if (value > maxes[3]) { maxes[3] = value }
          if (value < mins[3]) { mins[3] = value }
        }
        console.log(
          `x [${mins[0]}, ${maxes[0]}]
          y: [${mins[1]}, ${maxes[1]}]
          z: [${mins[2]}, ${maxes[2]}]
          value: [${mins[3]}, ${maxes[3]}]`
        );
        let min = 10;
        let max = -10;
        for (let i = 0; i < 3; i++) {
          if (mins[i] < min) { min = mins[i] }
          if (maxes[i] > max) { max = maxes[i] }
        }
        for (let i = 0; i < pointData.length; i+= 4) {
          pointData[i] = (pointData[i] - min) / (max - min)
          pointData[i + 1] = (pointData[i + 1] - min) / (max - min)
          pointData[i + 2] = (pointData[i + 2] - min) / (max - min)
          pointData[i + 3] = (pointData[i + 3] - mins[3]) / (maxes[3] - mins[3])
        }
        this.points = new Float32Array(pointData);
      
        // Parse the tetrahedra data
        const tetrahedraData: number[] = [];
        for (let i = this.numPoints + 1; i <= this.numPoints + this.numTetra; i++) {
          const idxs = lines[i].trim().split(' ').map(Number);
          idxs.sort(function (a, b) {
            return pointData[b * 4] - pointData[a * 4]
          });
          idxs.sort(function (a, b) {
            return pointData[b * 4 + 1] - pointData[a * 4 + 1]
          });
          idxs.sort(function (a, b) {
            return pointData[b * 4 + 2] - pointData[a * 4 + 2]
          });
          tetrahedraData.push(idxs[0], idxs[1], idxs[2], idxs[3]);
        }
        this.tetra = new Uint32Array(tetrahedraData);    
    }
}