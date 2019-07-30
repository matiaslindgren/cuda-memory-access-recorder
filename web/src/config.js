"use strict";

function failHard() {
	drawing = false;
	const errorBanner = document.getElementById("body-error-banner");
	errorBanner.innerHTML = "Something went wrong, please see the developer console";
	errorBanner.hidden = false;
	cancelDraw();
}

function assert(expr, msg, state) {
	if (!expr) {
		failHard();
		console.error("ASSERTION FAILED");
		if (typeof state !== "undefined") {
			console.error(state.name + " was:");
			printobj(state.obj);
		}
		throw "AssertionError: " + msg;
	}
}

function printobj(o) {
	console.log(JSON.stringify(o, null, 2));
}

//FIXME, maybe not needed anymore
function get4Palette(key) {
    let alpha = 0.2;
    let palette = [];
    switch(key) {
        case "rgba-colorful":
            palette = [
                [35, 196, 1, alpha],
                [227, 1, 23, alpha],
                [235, 190, 2, alpha],
                [47, 21, 182, alpha],
            ];
            break;
        case "rgba-grayscale":
            alpha = 0.5;
            palette = [
                [100, 100, 100, alpha],
                [200, 200, 200, alpha],
                [0, 0, 0, alpha],
                [50, 50, 50, alpha],
            ];
            break;
        default:
            failHard();
            console.error("unknown palette key: " + key);
    }
    return palette;
}


const CONFIG = {
	animation: {
		// Array of distinct colors to distinguish independent streaming multiprocessors
		SMColorPalette: get4Palette("rgba-colorful"),
		// SMColorPalette: get4Palette("rgba-grayscale"),
	},
	memory: {
		// Empty space between each slot on all sides
		slotPadding: 1,
		slotSizes: {
			min: 8,
			max: 36,
			step: 4,
		},
		//FIXME from submitted file
		rows: 32,
		columns: 32,
		slotFillRGBA: [160, 160, 160, 0.2],
		// Amount of animation steps of the cooldown transition after touching a memory index
		coolDownPeriod: 20,
	},
	SM: {
		count: {
			min: 1,
			max: 1,
		},
	},
};
