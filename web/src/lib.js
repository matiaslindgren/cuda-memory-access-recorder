class Drawable {
	// Simple HTML5 canvas primitive with dimensions, position and color
	constructor(label, x, y, width, height, canvas, fillRGBA) {
		this.label = label;
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.canvasContext = canvas.getContext("2d");
		this.changeFill((typeof fillRGBA === "undefined") ? [0, 0, 0, 0] : fillRGBA.slice());
	}

	changeFill(fillRGBA) {
		this.fillRGBA = fillRGBA.slice();
		this.fillStyle = "rgba(" + fillRGBA.join(',') + ')';
	}

	draw() {
		const ctx = this.canvasContext;
		ctx.fillStyle = this.fillStyle;
		ctx.fillRect(this.x, this.y, this.width, this.height);
	}
}

class DeviceMemory extends Drawable {
	// Global device memory space represented as a matrix of squares,
	// where each square is a single memory address.
	constructor(canvas, memoryColumnCount, memoryRowCount, slotSize, accessPatterns) {
		const rows = memoryRowCount;
		const columns = memoryColumnCount;
		const slotPadding = CONFIG.memory.slotPadding;
		const width = columns * slotSize + slotPadding * columns;
		let height = rows * slotSize + slotPadding * rows;
		const x = 0;
		const y = 0;
		super("device-memory", x, y, width, height, canvas);
		this.accessPatterns = accessPatterns;
		this.maxCycle = Math.max.apply(null, Array.from(this.accessPatterns.keys()));
		/* assert(typeof this.maxCycle !== "undefined", "Possibly invalid memory access pattern Map provided to DeviceMemory"); */
		const slotFillRGBA = CONFIG.memory.slotFillRGBA.slice();
		this.slots = Array.from(
			new Array(columns * rows),
			(_, i) => {
				const slotX = x + (i % columns) * (slotSize + slotPadding);
				const rowIndex = Math.floor(i / columns);
				let slotY = y + rowIndex * (slotSize + slotPadding);
				// Drawable memory slot
				const memorySlot = new MemorySlot(i, 2, "memory-slot", slotX, slotY, slotSize, slotSize, canvas, slotFillRGBA);
				// Color of the memory slot after a memory access
				const touchedColor = CONFIG.memory.accessedSlotColor.slice();
				const coolDownPeriod = CONFIG.memory.coolDownPeriod;
				// Alpha value decrement towards the final alpha after a memory access
				const coolDownStep = Math.abs(memorySlot.fillRGBA[3] - touchedColor[3]) / (coolDownPeriod + 1);
				return {
					memory: memorySlot,
					touchedColor: touchedColor,
					coolDownPeriod: coolDownPeriod,
					coolDownStep: coolDownStep,
					hotness: 0
				};
			}
		);
	}

	touch(memoryIndex) {
		const slot = this.slots[memoryIndex];
		slot.hotness = slot.coolDownPeriod;
		slot.memory.changeFill(slot.touchedColor);
	}

	programTerminated(cycle) {
		return cycle > this.maxCycle;
	}

	step(cycle) {
		// Get set of indexes that were accessed at a given SM cycle
		let accesses = this.accessPatterns.get(cycle) || new Set();
		for (let i of accesses) {
			this.touch(i);
		}
		for (let slot of this.slots) {
			slot.memory.draw();
		}
		this.draw();
	}

	draw() {
		for (let slot of this.slots) {
			if (slot.hotness > 0) {
				// The memory slot is still cooling down from a recent memory access
				--slot.hotness;
				let newFill = slot.memory.fillRGBA.slice();
				newFill[3] -= slot.coolDownStep;
				slot.memory.changeFill(newFill);
			}
		}
		super.draw();
	}

	clear() {
		for (let slot of this.slots) {
			slot.memory.clear();
			slot.hotness = 0;
		}
	}
}

class MemorySlot extends Drawable {
	// One memory slot represents a single address holding a 4-byte word
	constructor(index, value, ...drawableArgs) {
		super(...drawableArgs);
		this.index = index;
		this.value = value;
		this.defaultColor = this.fillRGBA.slice();
	}

	draw() {
		// Draw memory slot rectangle
		super.draw();
	}

	clear() {
		this.changeFill(his.defaultColor);
	}
}
