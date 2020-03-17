import * as tf from '@tensorflow/tfjs';
import 'babel-polyfill';

var canvas;
var model;
var tc;

const loadPage = () => {
	canvas = new fabric.Canvas('canvas', {
		backgroundColor: '#ffffff',
		width: 300,
		height: 300,
		isDrawingMode: true
	});
	tc = document.getElementById('tc')
	// tc = new fabric.Canvas('tc')
	canvas.freeDrawingBrush.width = 10;
	canvas.freeDrawingBrush.color = 'black';
	canvas.on('mouse:up', predictSymbol);
	canvas.renderAll();
	loadModel();
	var clsBtn = document.getElementById('clearBtn');
	clsBtn.onclick = clearCanvas;
};

const clearCanvas = () => {
	canvas.clear();
	canvas.backgroundColor = '#ffffff';
};

const loadModel = async () => {
	model = await tf.loadLayersModel('http://localhost:8080/model.json');
	console.log('Model loaded');
	document.getElementById('load').innerHTML = 'Model loaded.';
};

const preprocessData = (imageData) => {
	return tf.tidy(() => {
		let imageTensor = tf.browser.fromPixels(imageData, 1);
		let resizedImage = tf.image.resizeBilinear(imageTensor, [ 32, 32 ]).toFloat();

		// normalize image
		const normConst = tf.scalar(255.0);
		let normalizedImage = resizedImage.div(normConst);

		// Add a dimension to be suitable for prediction
		normalizedImage = normalizedImage.expandDims(0);
		return normalizedImage;
	});
};

const predictSymbol = () => {
	const testData = getImage();
	// let probabilities = model.predict(testData).dataSync();
	// let topK = findIndicesOfTopK(probabilities, 5);
	// let topKProb = findTopKProb(probabilities, topK);
	// console.log('topK:', topK);
	// console.log('topKProb:', topKProb);
};

const findTopKProb = (pr, topK) => {
	let topKProb = [];
	for (let index = 0; index < topK.length; index++) {
		topKProb.push(pr[topK[index]]);
	}
	return topKProb;
};

const findIndicesOfTopK = (pr, k) => {
	var result = [];
	for (var i = 0; i < pr.length; i++) {
		result.push(i); // add index to output array
		if (result.length > k) {
			result.sort(function(a, b) {
				return pr[b] - pr[a];
			}); // descending sort the output array
			result.pop(); // remove the last index (index of smallest element in output array)
		}
	}
	return result;
};

const getImage = () => {
	canvas.renderAll();
	let ctx = document.getElementById('canvas').getContext('2d')

		// let ctx = canvas.getContext();
	const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
	// let tc = document.getElementById('tc')
	ctx.clearRect(0,0,canvas.width, canvas.height)
	ctx.canvas.width = 300
	ctx.canvas.height = 300
	let tctx = tc.getContext('2d');
	canvas.width = 300
	canvas.height = 300
	tctx.putImageData(imageData,0,0)
	ctx.putImageData(imageData,0,0)
	// canvas.add(i)
	// canvas.renderAll()
	console.log(canvas.getObjects())
	console.log('done')
	// return preprocessData(imageData);
};

window.onload = loadPage;
