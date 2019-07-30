"use strict";

// All mutable global variables
var memoryCanvasInput;
var device;
var drawing = false;
var prevRenderFrameID;
var memorySlotSize = 20;
var accessPatterns = new Map;
var currentCycle = 0;


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

// Define menubar buttons for interacting with the animation
function initUI() {
	const pauseButton = document.getElementById("pause-button");
	const restartButton = document.getElementById("restart-button");
	const memorySlotSizeSelect = document.getElementById("memory-slot-size-select");
	const accessPatternFileInput = document.getElementById("access-patterns-file");
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
			for (let line of content.split("\n").slice(1)) {
				let [index, cycle] = line.split('\t').map(x => parseInt(x));
				if (typeof accessPatterns.get(cycle) === "undefined") {
					accessPatterns.set(cycle, new Set([index]));
				} else {
					accessPatterns.get(cycle).add(index);
				}
			}
			console.log("Parsed", accessPatterns.size, "memory access events");
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
	device = new DeviceMemory(memoryCanvasInput, memorySlotSize, accessPatterns);
	// Resize memory canvas and its container depending on the input array size
	const memoryCanvasContainer = document.getElementById("memoryCanvasContainer");
	const canvasWidth = CONFIG.memory.columns * (CONFIG.memory.slotPadding + memorySlotSize);
	let canvasHeight = CONFIG.memory.rows * (CONFIG.memory.slotPadding + memorySlotSize);
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
	device.step(currentCycle++);
	if (currentCycle % 100 == 0) {
		console.log("cycle", currentCycle);
	}
	if (device.programTerminated(currentCycle)) {
		device.clear();
		clear(memoryCanvasInput, "hard");
		device.step(-1);
		drawing = false;
	}
	if (drawing) {
		queueDraw();
	}
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
