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
		assert([x, y, width, height, ctx].every(val => typeof val !== "undefined"), "Drawable instances must always have defined x, y, width, height, and canvasContext.", {name: "Drawable", obj: this});
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
	constructor(canvas, slotSize, accessPatterns) {
		const rows = CONFIG.memory.rows;
		const columns = CONFIG.memory.columns;
		const slotPadding = CONFIG.memory.slotPadding;
		const width = columns * slotSize + slotPadding * columns;
		let height = rows * slotSize + slotPadding * rows;
		const x = 0;
		const y = 0;
		super("device-memory", x, y, width, height, canvas);
		this.accessPatterns = accessPatterns;
		this.maxCycle = Math.max.apply(null, Array.from(this.accessPatterns.keys()));
		assert(typeof this.maxCycle !== "undefined", "Possibly invalid memory access pattern Map provided to DeviceMemory");
		const slotFillRGBA = CONFIG.memory.slotFillRGBA.slice();
		this.slots = Array.from(
			new Array(columns * rows),
			(_, i) => {
				const slotX = x + (i % columns) * (slotSize + slotPadding);
				const rowIndex = Math.floor(i / columns);
				let slotY = y + rowIndex * (slotSize + slotPadding);
				// Drawable memory slot
				const memorySlot = new MemorySlot(i, 2, "memory-slot", slotX, slotY, slotSize, slotSize, canvas, undefined, slotFillRGBA);
				// Drawable overlays of different colors on top of the slot, one for each SM
				const overlays = Array.from(
					CONFIG.animation.SMColorPalette,
					SM_color => {
						const coolDownPeriod = CONFIG.memory.coolDownPeriod;
						const coolDownStep = (1.0 - SM_color[3]) / (coolDownPeriod + 1);
						return {
							drawable: new Drawable("memory-slot-overlay-SM-color", slotX, slotY, slotSize, slotSize, canvas, undefined, SM_color),
							defaultColor: SM_color.slice(),
							hotness: 0,
							coolDownPeriod: coolDownPeriod,
							coolDownStep: coolDownStep,
						};
					}
				);
				// Counter indexed by SM ids, counting how many threads of that SM is currently waiting for a memory access to complete from this memory slot i.e. DRAM index
				let threadAccessCounter = new Array(CONFIG.SM.count.max);
				threadAccessCounter.fill(0);
				return {
					memory: memorySlot,
					overlays: overlays,
					threadAccessCounter: threadAccessCounter,
				};
			}
		);
	}

	// Simulated memory access to DRAM index `memoryIndex` by an SM with id `SM_ID`
	touch(SM_ID, memoryIndex) {
		assert(typeof memoryIndex !== "undefined", "memoryIndex must be defined when touching memory slot");
		assert(typeof SM_ID !== "undefined", "SM_ID must be defined when touching memory slot");
		assert(CONFIG.SM.count.min <= SM_ID <= CONFIG.SM.count.max, "attempting to touch a DRAM index " + memoryIndex + " with multiprocessor ID " + SM_ID + " which is out of range of minimum and maximum amount of SMs");
		const slot = this.slots[memoryIndex];
		++slot.threadAccessCounter[SM_ID - 1];
		let overlay = slot.overlays[SM_ID - 1];
		overlay.hotness = overlay.coolDownPeriod;
		overlay.drawable.fillRGBA[3] = 0.6;
	}

	programTerminated(cycle) {
		return cycle > this.maxCycle;
	}

	step(cycle) {
		// Get set of indexes that were accessed at a given SM cycle
		let accesses = this.accessPatterns.get(cycle) || new Set();
		for (let [i, slot] of this.slots.entries()) {
			if (accesses.has(i)) {
				this.touch(1, i);
			}
			slot.memory.draw();
		}
		this.draw();
	}

	// Assuming SM_ID integers in range(1, CONFIG.SM.count.max + 1),
	// Generator that yields the SM_IDs of corresponding indexes of all non-zero thread access counters
	*SMsCurrentlyAccessing(slot) {
		const counter = slot.threadAccessCounter;
		for (let SM_ID = 1; SM_ID < counter.length + 1; ++SM_ID) {
			if (counter[SM_ID - 1] > 0) {
				yield SM_ID;
			}
		}
	}

	numSMsCurrentlyAccessing(slot) {
		return Array.from(this.SMsCurrentlyAccessing(slot)).length;
	}

	draw() {
		for (let [i, slot] of this.slots.entries()) {
			// On top of the memory slot, draw unique color for each SM currently accessing this memory slot
			let SM_count = 0;
			const numSMs = this.numSMsCurrentlyAccessing(slot);
			for (let SM_ID = 1; SM_ID < slot.overlays.length + 1; ++SM_ID) {
				let overlay = slot.overlays[SM_ID - 1];
				const drawable = overlay.drawable;
				assert(typeof drawable !== "undefined", "All memory slots must have drawables for every SM overlay color");
				if (slot.threadAccessCounter[SM_ID - 1] > 0) {
					// Some thread of a warp scheduled in this SM is still accessing memory slot i
					// Draw small slice of original so that all slices fit in the slot
					SM_count++;
					const originalX = drawable.x;
					const originalWidth = drawable.width;
					drawable.x += (numSMs - SM_count) * originalWidth / numSMs;
					drawable.width /= numSMs;
					drawable.draw();
					// Put back original size
					drawable.x = originalX;
					drawable.width = originalWidth;
					// Do not reduce hotness too much
					if (overlay.hotness > overlay.coolDownPeriod/2) {
						// Hotness still half way
						--overlay.hotness;
						overlay.drawable.fillRGBA[3] -= overlay.coolDownStep;
					}
				} else {
					// No threads are accessing this slot
					if (overlay.hotness > 0) {
						// Overlay still has alpha to display SM color
						--overlay.hotness;
						drawable.draw();
					}
					if (overlay.hotness === 0) {
						// Set alpha to zero to remove SM color
						overlay.drawable.fillRGBA[3] = 0;
					} else {
						// Reduce alpha slightly
						overlay.drawable.fillRGBA[3] -= overlay.coolDownStep;
					}
				}
			}
		}
		super.draw();
	}

	clear() {
		for (let slot of this.slots) {
			slot.memory.clear();
			slot.threadAccessCounter.fill(0);
			slot.overlays.forEach(o => o.hotness = 0);
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
		// Then, draw memory access progress on top of it
		const x = this.x;
		const y = this.y;
		const width = this.width;
		const height = this.height;
		const ctx = this.canvasContext;
	}

	clear() {
		this.fillRGBA = this.defaultColor.slice();
	}
}
