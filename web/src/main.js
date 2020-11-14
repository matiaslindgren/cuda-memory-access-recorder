"use strict";

// All mutable global variables
var memoryCanvasInput;
var device;
var drawing = false;
var prevRenderFrameID;
var memorySlotSize = 8;
var accessPatterns;
var currentCycle = 0;
var memoryRowCount = 64;
var memoryColumnCount = 64;
var cycleCounterSpan;
var numberOfSMs;
var stepsPerCycle = 100;


function makeMemorySlotSizeSelectOptionsHTML(config) {
	function makeOption(key) {
		return '<option value="' + key + '"' + ((key === memorySlotSize) ? 'selected' : '') + '>DRAM slot size ' + key + '</option>';
	}
	const options = [];
	for (let size = config.min; size <= config.max; size += config.step) {
		options.push(makeOption(size));
	}
	return options.join("\n");
}

function resetSize(element, newWidth, newHeight) {
	element.width = newWidth;
	element.height = newHeight;
	element.style.width = element.width.toString() + "px";
	element.style.height = element.height.toString() + "px";
}

// Group all adjacent accesses into groups, each of size 'stepsPerCycle'
// In the visualization, these accesses will seem to happen in the same time
// This increases performance of the animation loop
function groupAccessPatterns(accessPatterns, stepsPerCycle) {
	const accessPatternGroups = new Map;
	const allCycles = Array.from(accessPatterns.keys());
	let maxCycle = -1;
	for (let c of allCycles) {
		maxCycle = Math.max(c, maxCycle);
	}
	for (let cycle = 0; cycle <= maxCycle + stepsPerCycle; cycle += stepsPerCycle) {
		const groupIndexes = new Set;
		for (let c = 0; c < stepsPerCycle; ++c) {
			const indexes = accessPatterns.get(cycle + c) || new Set;
			for (let i of indexes) {
				groupIndexes.add(i);
			}
		}
		accessPatternGroups.set(cycle, groupIndexes);
	}
	return accessPatternGroups;
}

// Initialize access patterns from submitted JSON file
function JSONFileHandler(event) {
	let animationDefinition = JSON.parse(event.target.result);
	numberOfSMs.innerHTML = animationDefinition.num_SMs_used || "n/a";
	memoryColumnCount = animationDefinition.cols;
	memoryRowCount = animationDefinition.rows;
	if (animationDefinition.accesses.length !== animationDefinition.clocks.length) {
		console.error("accesses and clocks arrays should be of equal length since their elements are defined as pairs.");
	}
	accessPatterns = new Map;
	for (let i = 0; i < animationDefinition.accesses.length; ++i) {
		const cycle = animationDefinition.clocks[i];
		const index = animationDefinition.accesses[i];
		if (typeof accessPatterns.get(cycle) === "undefined") {
			accessPatterns.set(cycle, new Set([index]));
		} else {
			accessPatterns.get(cycle).add(index);
		}
	}
	console.log("Parsed", accessPatterns.size, "memory access events");
	if (stepsPerCycle > 1) {
		accessPatterns = groupAccessPatterns(accessPatterns, stepsPerCycle);
	}
	restart();
};

// Define menubar buttons for interacting with the animation
function initUI() {
	const pauseButton = document.getElementById("pause-button");
	const restartButton = document.getElementById("restart-button");
	const memorySlotSizeSelect = document.getElementById("memory-slot-size-select");
	const accessPatternFileInput = document.getElementById("access-patterns-file");
	cycleCounterSpan = document.getElementById("cycle-counter");
	numberOfSMs = document.getElementById("num-SMs");
	pauseButton.addEventListener("click", _ => {
		pause();
		pauseButton.value = drawing ? "Pause" : "Continue";
	});
	restartButton.addEventListener("click", _ => {
		pauseButton.value = "Pause";
		restart();
	});
	memorySlotSizeSelect.addEventListener("change", event => {
		drawing = false;
		memorySlotSize = parseInt(event.target.value);
		pauseButton.value = "Pause";
		restart();
	});
	accessPatternFileInput.addEventListener("change", event => {
		const file = event.target.files[0];
		console.log("Parsing", file.name, "of size", file.size);
		const reader = new FileReader();
		reader.onload = JSONFileHandler;
		reader.readAsText(file);
	});
}

function populateUI() {
	document.getElementById("memory-slot-size-select").innerHTML = makeMemorySlotSizeSelectOptionsHTML(CONFIG.memory.slotSizes);
}

function initSimulation() {
	memoryCanvasInput = document.getElementById("memoryCanvasInput");
	// Initialize simulated GPU
	device = new DeviceMemory(memoryCanvasInput, memoryColumnCount, memoryRowCount, memorySlotSize, accessPatterns);
	// Resize memory canvas and its container depending on the input array size
	const memoryCanvasContainer = document.getElementById("memoryCanvasContainer");
	const canvasWidth = memoryColumnCount * (CONFIG.memory.slotPadding + memorySlotSize);
	let canvasHeight = memoryRowCount * (CONFIG.memory.slotPadding + memorySlotSize);
	resetSize(memoryCanvasInput, canvasWidth, canvasHeight);
	resetSize(memoryCanvasContainer, canvasWidth, memoryCanvasContainer.height + canvasHeight);
	currentCycle = 0;
}

function clear(canvas, type) {
	const ctx = canvas.getContext("2d");
	ctx.fillStyle = 'rgba(255, 255, 255, ' + ((type === "hard") ? 1.0 : 0.7) + ')';
	ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function cancelDraw() {
	window.cancelAnimationFrame(prevRenderFrameID);
}

function queueDraw() {
	prevRenderFrameID = window.requestAnimationFrame(draw);
}

function draw(now) {
	clear(memoryCanvasInput, "hard");
	device.step(currentCycle);
	cycleCounterSpan.innerHTML = currentCycle;
	if (device.programTerminated(currentCycle)) {
		device.clear();
		clear(memoryCanvasInput, "hard");
		device.step(-1);
		drawing = false;
	}
	if (drawing) {
		queueDraw();
	}
	currentCycle += stepsPerCycle;
}

function restart() {
	console.log("Starting animation");
	drawing = false;
	cancelDraw();
	initSimulation();
	drawing = true;
	queueDraw();
}

function pause() {
	drawing = !drawing;
	if (drawing) {
		queueDraw();
	}
}

document.addEventListener("DOMContentLoaded", _ => {
	initUI();
	populateUI();
});
