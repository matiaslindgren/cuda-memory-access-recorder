"use strict";

// All mutable global variables
var memoryCanvasInput;
var device;
var drawing = false;
var prevRenderFrameID;
var memorySlotSize = 20;
var accessPatterns = new Map;
var currentCycle = 0;
var memoryRowCount = 64;
var memoryColumnCount = 64;
var cycleCounterSpan;
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
	const maxCycle = Math.max.apply(null, allCycles);
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

// Define menubar buttons for interacting with the animation
function initUI() {
	const pauseButton = document.getElementById("pause-button");
	const restartButton = document.getElementById("restart-button");
	const memorySlotSizeSelect = document.getElementById("memory-slot-size-select");
	const accessPatternFileInput = document.getElementById("access-patterns-file");
	cycleCounterSpan = document.getElementById("cycle-counter");
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
		//FIXME validate file
		const file = event.target.files[0];
		console.log("Parsing", file.name, "of size", file.size);
		const filePromise = new Response(file.slice(0, file.size - 1)).text();
		filePromise.then(content => {
			const lines = content.split('\n');
			for (let line of lines.slice(1)) {
				let [index, cycle] = line.split('\t').map(x => parseInt(x));
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
		});
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
