class Drawable {
	// Simple HTML5 canvas primitive with dimensions, position and color
	constructor(label, x, y, width, height, canvas, strokeRGBA, fillRGBA) {
		this.label = label;
		this.x = x;
		this.y = y;
		this.width = width;
		this.height = height;
		this.canvasContext = canvas.getContext("2d");
		// Copy RGBA arrays, if given, else set to null
		this.strokeRGBA = (typeof strokeRGBA === "undefined") ? null : strokeRGBA.slice();
		this.fillRGBA = (typeof fillRGBA === "undefined") ? null : fillRGBA.slice();
	}

	draw() {
		const x = this.x;
		const y = this.y;
		const width = this.width;
		const height = this.height;
		const ctx = this.canvasContext;
		/* assert([x, y, width, height, ctx].every(val => typeof val !== "undefined"), "Drawable instances must always have defined x, y, width, height, and canvasContext.", {name: "Drawable", obj: this}); */
		if (this.fillRGBA !== null) {
			ctx.fillStyle = "rgba(" + this.fillRGBA.join(',') + ')';
			ctx.fillRect(x, y, width, height);
		}
		if (this.strokeRGBA !== null) {
			ctx.strokeStyle = "rgba(" + this.strokeRGBA.join(',') + ')';
			ctx.strokeRect(x, y, width, height);
		}
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
				const memorySlot = new MemorySlot(i, 2, "memory-slot", slotX, slotY, slotSize, slotSize, canvas, undefined, slotFillRGBA);
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
		slot.memory.fillRGBA = slot.touchedColor.slice();
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
				slot.memory.fillRGBA[3] -= slot.coolDownStep;
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
		this.fillRGBA = this.defaultColor.slice();
	}
}
