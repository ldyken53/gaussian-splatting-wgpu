interface Point {
    x: number;
    y: number;
}

interface Margin {
    top: number;
    right: number;
    bottom: number;
    left: number;
}

type OpacityUpdateCallback = (buffer: number[]) => void;

export class OpacityTransferFunction {
    private canvas: HTMLCanvasElement;
    private ctx: CanvasRenderingContext2D;
    private width: number;
    private height: number;
    private margin: Margin;
    private graphWidth: number;
    private graphHeight: number;
    private controlPoints: Point[];
    private dragging: boolean;
    private dragIndex: number;
    private pointRadius: number;
    private onUpdateCallback: OpacityUpdateCallback | null;

    constructor(onUpdate?: OpacityUpdateCallback) {
        this.onUpdateCallback = onUpdate;

        // Find elements by ID
        this.canvas = document.getElementById('opacityCanvas') as HTMLCanvasElement;
        if (!this.canvas) {
            throw new Error('Canvas element with id "opacityCanvas" not found');
        }

        const context = this.canvas.getContext('2d');
        if (!context) {
            throw new Error('Could not get 2D context from canvas');
        }
        this.ctx = context;
        
        this.width = this.canvas.width;
        this.height = this.canvas.height;
        
        // Margins for the graph
        this.margin = { top: 15, right: 15, bottom: 25, left: 35 };
        this.graphWidth = this.width - this.margin.left - this.margin.right;
        this.graphHeight = this.height - this.margin.top - this.margin.bottom;
        
        // Control points
        this.controlPoints = [
            { x: 0.0, y: 0.0 },
            { x: 1.0, y: 1.0 }
        ];
        
        // Interaction state
        this.dragging = false;
        this.dragIndex = -1;
        this.pointRadius = 4;
        
        this.setupEventListeners();
        this.setupButtonEventListeners();
        this.draw();
        this.updateOutput();
    }

    public setUpdateCallback(onUpdate: OpacityUpdateCallback) {
        this.onUpdateCallback = onUpdate;
        this.updateOutput();
    }
    
    private setupEventListeners(): void {
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('contextmenu', (e) => this.onRightClick(e));
    }
    
    private setupButtonEventListeners(): void {
        const clearBtn = document.getElementById('clearOpacityBtn');
        const resetBtn = document.getElementById('resetOpacityBtn');
        const getBtn = document.getElementById('getOpacityBtn');
        
        if (clearBtn) {
            clearBtn.addEventListener('click', () => this.clear());
        }
        
        if (resetBtn) {
            resetBtn.addEventListener('click', () => this.reset());
        }
        
        if (getBtn) {
            getBtn.addEventListener('click', (e) => this.getFunctionAndCopy(e));
        }
    }
    
    private getMousePos(e: MouseEvent): Point {
        const rect = this.canvas.getBoundingClientRect();
        return {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
    }
    
    private screenToGraph(screenX: number, screenY: number): Point {
        const x = (screenX - this.margin.left) / this.graphWidth;
        const y = 1.0 - (screenY - this.margin.top) / this.graphHeight;
        return { 
            x: Math.max(0, Math.min(1, x)), 
            y: Math.max(0, Math.min(1, y)) 
        };
    }
    
    private graphToScreen(x: number, y: number): Point {
        const screenX = this.margin.left + x * this.graphWidth;
        const screenY = this.margin.top + (1.0 - y) * this.graphHeight;
        return { x: screenX, y: screenY };
    }
    
    private findNearestPoint(screenX: number, screenY: number): number {
        for (let i = 0; i < this.controlPoints.length; i++) {
            const screen = this.graphToScreen(this.controlPoints[i].x, this.controlPoints[i].y);
            const distance = Math.sqrt(
                Math.pow(screenX - screen.x, 2) + Math.pow(screenY - screen.y, 2)
            );
            if (distance <= this.pointRadius + 3) {
                return i;
            }
        }
        return -1;
    }
    
    private onMouseDown(e: MouseEvent): void {
        const mousePos = this.getMousePos(e);
        const pointIndex = this.findNearestPoint(mousePos.x, mousePos.y);
        
        if (pointIndex !== -1) {
            this.dragging = true;
            this.dragIndex = pointIndex;
            this.canvas.style.cursor = 'grabbing';
        } else {
            // Add new point
            const graphPos = this.screenToGraph(mousePos.x, mousePos.y);
            this.controlPoints.push(graphPos);
            this.controlPoints.sort((a, b) => a.x - b.x);
            this.draw();
            this.updateOutput();
        }
    }
    
    private onMouseMove(e: MouseEvent): void {
        const mousePos = this.getMousePos(e);
        
        if (this.dragging && this.dragIndex !== -1) {
            const graphPos = this.screenToGraph(mousePos.x, mousePos.y);
            this.controlPoints[this.dragIndex] = graphPos;
            
            // Sort points by x coordinate but keep track of the moved point
            const movedPoint = this.controlPoints[this.dragIndex];
            this.controlPoints.sort((a, b) => a.x - b.x);
            this.dragIndex = this.controlPoints.indexOf(movedPoint);
            
            this.draw();
            this.updateOutput();
        } else {
            // Update cursor based on hover
            const pointIndex = this.findNearestPoint(mousePos.x, mousePos.y);
            this.canvas.style.cursor = pointIndex !== -1 ? 'grab' : 'crosshair';
        }
    }
    
    private onMouseUp(e: MouseEvent): void {
        this.dragging = false;
        this.dragIndex = -1;
        this.canvas.style.cursor = 'crosshair';
    }
    
    private onRightClick(e: MouseEvent): void {
        e.preventDefault();
        const mousePos = this.getMousePos(e);
        const pointIndex = this.findNearestPoint(mousePos.x, mousePos.y);
        
        if (pointIndex !== -1 && this.controlPoints.length > 2) {
            this.controlPoints.splice(pointIndex, 1);
            this.draw();
            this.updateOutput();
        }
    }
    
    private interpolate(x: number): number {
        if (this.controlPoints.length === 0) return 0;
        if (this.controlPoints.length === 1) return this.controlPoints[0].y;
        
        // Find the two points to interpolate between
        let leftPoint = this.controlPoints[0];
        let rightPoint = this.controlPoints[this.controlPoints.length - 1];
        
        for (let i = 0; i < this.controlPoints.length - 1; i++) {
            if (x >= this.controlPoints[i].x && x <= this.controlPoints[i + 1].x) {
                leftPoint = this.controlPoints[i];
                rightPoint = this.controlPoints[i + 1];
                break;
            }
        }
        
        // Linear interpolation
        if (leftPoint.x === rightPoint.x) {
            return leftPoint.y;
        }
        
        const t = (x - leftPoint.x) / (rightPoint.x - leftPoint.x);
        return leftPoint.y + t * (rightPoint.y - leftPoint.y);
    }
    
    public generateOpacityBuffer(): number[] {
        const buffer = new Array<number>(100);
        for (let i = 0; i < 100; i++) {
            const x = i / 99.0; // 0.0 to 1.0
            buffer[i] = this.interpolate(x);
        }
        return buffer;
    }
    
    private draw(): void {
        // Clear canvas
        this.ctx.clearRect(0, 0, this.width, this.height);
        
        // Draw background grid
        this.drawGrid();
        
        // Draw function curve
        this.drawCurve();
        
        // Draw control points
        this.drawControlPoints();
        
        // Draw axes
        this.drawAxes();
    }
    
    private drawGrid(): void {
        this.ctx.strokeStyle = '#e9ecef';
        this.ctx.lineWidth = 1;
        
        // Vertical grid lines
        for (let i = 0; i <= 5; i++) {
            const x = this.margin.left + (i / 5) * this.graphWidth;
            this.ctx.beginPath();
            this.ctx.moveTo(x, this.margin.top);
            this.ctx.lineTo(x, this.margin.top + this.graphHeight);
            this.ctx.stroke();
        }
        
        // Horizontal grid lines
        for (let i = 0; i <= 5; i++) {
            const y = this.margin.top + (i / 5) * this.graphHeight;
            this.ctx.beginPath();
            this.ctx.moveTo(this.margin.left, y);
            this.ctx.lineTo(this.margin.left + this.graphWidth, y);
            this.ctx.stroke();
        }
    }
    
    private drawCurve(): void {
        this.ctx.strokeStyle = '#007bff';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        
        for (let i = 0; i < this.graphWidth; i++) {
            const x = i / this.graphWidth;
            const y = this.interpolate(x);
            const screen = this.graphToScreen(x, y);
            
            if (i === 0) {
                this.ctx.moveTo(screen.x, screen.y);
            } else {
                this.ctx.lineTo(screen.x, screen.y);
            }
        }
        this.ctx.stroke();
    }
    
    private drawControlPoints(): void {
        this.controlPoints.forEach((point) => {
            const screen = this.graphToScreen(point.x, point.y);
            
            // Point circle
            this.ctx.fillStyle = '#007bff';
            this.ctx.strokeStyle = 'white';
            this.ctx.lineWidth = 1;
            this.ctx.beginPath();
            this.ctx.arc(screen.x, screen.y, this.pointRadius, 0, 2 * Math.PI);
            this.ctx.fill();
            this.ctx.stroke();
        });
    }
    
    private drawAxes(): void {
        this.ctx.strokeStyle = '#495057';
        this.ctx.lineWidth = 1;
        this.ctx.font = '8px Arial';
        this.ctx.fillStyle = '#495057';
        
        // X-axis
        this.ctx.beginPath();
        this.ctx.moveTo(this.margin.left, this.margin.top + this.graphHeight);
        this.ctx.lineTo(this.margin.left + this.graphWidth, this.margin.top + this.graphHeight);
        this.ctx.stroke();
        
        // Y-axis
        this.ctx.beginPath();
        this.ctx.moveTo(this.margin.left, this.margin.top);
        this.ctx.lineTo(this.margin.left, this.margin.top + this.graphHeight);
        this.ctx.stroke();
        
        // X-axis labels
        this.ctx.fillText('0', this.margin.left - 3, this.margin.top + this.graphHeight + 12);
        this.ctx.fillText('1', this.margin.left + this.graphWidth - 3, this.margin.top + this.graphHeight + 12);
        
        // Y-axis labels
        this.ctx.fillText('0', this.margin.left - 12, this.margin.top + this.graphHeight + 3);
        this.ctx.fillText('1', this.margin.left - 12, this.margin.top + 3);
    }
    
    private updateOutput(): void {
        const buffer = this.generateOpacityBuffer();
        
        // Call the callback if provided
        if (this.onUpdateCallback) {
            this.onUpdateCallback(buffer);
        }
    }
    
    public clear(): void {
        this.controlPoints = [];
        this.draw();
        this.updateOutput();
    }
    
    public reset(): void {
        this.controlPoints = [
            { x: 0.0, y: 0.0 },
            { x: 1.0, y: 1.0 }
        ];
        this.draw();
        this.updateOutput();
    }
    
    public getFunction(): number[] {
        return this.generateOpacityBuffer();
    }
    
    private getFunctionAndCopy(e: Event): void {
        const buffer = this.getFunction();
        console.log('Opacity function buffer:', buffer);
        
        // Copy to clipboard
        navigator.clipboard.writeText(JSON.stringify(buffer)).then(() => {
            const btn = e.target as HTMLButtonElement;
            const originalText = btn.textContent || '';
            btn.textContent = 'Copied!';
            btn.classList.remove('btn-outline-primary');
            btn.classList.add('btn-success');
            setTimeout(() => {
                btn.textContent = originalText;
                btn.classList.remove('btn-success');
                btn.classList.add('btn-outline-primary');
            }, 1000);
        }).catch((err) => {
            console.error('Failed to copy to clipboard:', err);
        });
    }
}