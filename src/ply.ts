import { PackingType, StaticArray, Struct, vec3, vec4, f32 } from "./packing";

export function loadFileAsArrayBuffer(file: File): Promise<ArrayBuffer> {
    /* loads a file as an ArrayBuffer (i.e. a binary blob) */
    return new Promise((resolve, reject) => {
        const reader = new FileReader();

        reader.onload = (event) => {
            if (!event.target || !event.target.result) {
                reject('Failed to load file');
                return;
            }
            if (typeof event.target.result === 'string') {
                reject('Got a text file instead of a binary one');
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

        reader.readAsArrayBuffer(file);
    });
}

export class PackedGaussians {
    /* A class that
        1) reads the binary blob from a .ply file
        2) converts internally into a structured representation
        3) packs the structured representation into a flat array of bytes as expected by the shaders
    */
    numGaussians: number;
    sphericalHarmonicsDegree: number;

    gaussianLayout: PackingType;
    public gaussianArrayLayout: PackingType;
    positionsLayout: PackingType;
    public positionsArrayLayout: PackingType;

    gaussiansBuffer: ArrayBuffer;
    positionsBuffer: ArrayBuffer;

    private static decodeHeader(plyArrayBuffer: ArrayBuffer): [number, Record<string, string>, DataView] {
        /* decodes the .ply file header and returns a tuple of:
            * - vertexCount: number of vertices in the point cloud
            * - propertyTypes: a map from property names to their types
            * - vertexData: a DataView of the vertex data
        */

        const decoder = new TextDecoder();
        let headerOffset = 0;
        let headerText = '';

        while (true) {
            const headerChunk = new Uint8Array(plyArrayBuffer, headerOffset, 50);
            headerText += decoder.decode(headerChunk);
            headerOffset += 50;

            if (headerText.includes('end_header')) {
                break;
            }
        }

        const headerLines = headerText.split('\n');

        let vertexCount = 0;
        let propertyTypes: Record<string, string> = {};

        for (let i = 0; i < headerLines.length; i++) {
            const line = headerLines[i].trim();
            if (line.startsWith('element vertex')) {
                const vertexCountMatch = line.match(/\d+/);
                if (vertexCountMatch) {
                    vertexCount = parseInt(vertexCountMatch[0]);
                }
            } else if (line.startsWith('property')) {
                const propertyMatch = line.match(/(\w+)\s+(\w+)\s+(\w+)/);
                if (propertyMatch) {
                    const propertyType = propertyMatch[2];
                    const propertyName = propertyMatch[3];
                    propertyTypes[propertyName] = propertyType;
                }
            } else if (line === 'end_header') {
                break;
            }
        }

        const vertexByteOffset = headerText.indexOf('end_header') + 'end_header'.length + 1;
        const vertexData = new DataView(plyArrayBuffer, vertexByteOffset);

        return [
            vertexCount,
            propertyTypes,
            vertexData,
        ];
    }

    private readRawVertex(line: string, propertyTypes: Record<string, string>): Record<string, number> {
        /* reads a single vertex from a line of text and returns:
            * - rawVertex: a map from property names to their values
        */
        let rawVertex: Record<string, number> = {};
        const data = line.split(' ').map(Number);
        let i = 0;
        for (const property in propertyTypes) {
            rawVertex[property] = data[i];
            i++;
        }

        return rawVertex;
    }

    private arrangeVertex(rawVertex: Record<string, number>): Record<string, any> {
        /* arranges a raw vertex into a vertex that can be packed by the gaussianLayout utility */
        const arrangedVertex: Record<string, any> = {
            position: [rawVertex.x, rawVertex.y, rawVertex.z],
            value: rawVertex.value,
            logScale: [rawVertex.scale_0, rawVertex.scale_1, rawVertex.scale_2],
            rotQuat: [rawVertex.rot_0, rawVertex.rot_1, rawVertex.rot_2, rawVertex.rot_3],
        };
        return arrangedVertex;
    }

    constructor(arrayBuffer: ArrayBuffer) {
        // decode the header
        const [vertexCount, propertyTypes, vertexData] = PackedGaussians.decodeHeader(arrayBuffer);
        this.numGaussians = vertexCount;

        // define the layout of a single point
        this.gaussianLayout = new Struct([
            ['position', new vec3(f32)],
            ['value', f32],
            ['logScale', new vec3(f32)],
            ['rotQuat', new vec4(f32)],
        ]);
        // define the layout of the entire point cloud
        this.gaussianArrayLayout = new StaticArray(this.gaussianLayout, vertexCount);

        this.positionsLayout = new vec3(f32);
        this.positionsArrayLayout = new StaticArray(this.positionsLayout, vertexCount);

        // pack the points
        this.gaussiansBuffer = new ArrayBuffer(this.gaussianArrayLayout.size);
        const gaussianWriteView = new DataView(this.gaussiansBuffer);

        this.positionsBuffer = new ArrayBuffer(this.positionsArrayLayout.size);
        const positionsWriteView = new DataView(this.positionsBuffer);

        const decoder = new TextDecoder();
        const vertexText = decoder.decode(vertexData);
        const vertexLines = vertexText.split('\n');

        var gaussianWriteOffset = 0;
        var positionWriteOffset = 0;
        var minVal = 10;
        var maxVal = 0;
        for (let i = 0; i < vertexCount; i++) {
            const line = vertexLines[i].trim();
            const rawVertex = this.readRawVertex(line, propertyTypes);
            gaussianWriteOffset = this.gaussianLayout.pack(
                gaussianWriteOffset,
                this.arrangeVertex(rawVertex),
                gaussianWriteView,
            );

            positionWriteOffset = this.positionsLayout.pack(
                positionWriteOffset,
                [rawVertex.x, rawVertex.y, rawVertex.z],
                positionsWriteView,
            );
            if (rawVertex.value > maxVal) {
                maxVal = rawVertex.value;
            }
            if (rawVertex.value < minVal) {
                minVal = rawVertex.value;
            }
        }
        console.log(`Range: [${minVal}, ${maxVal}]`);
    }
}