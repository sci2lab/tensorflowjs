import * as tf from '@tensorflow/tfjs';
import 'babel-polyfill';
import $ from './jquery'


var canvas;
var model;
var symbols = [{}];

const loadPage = () => {
	canvas = new fabric.Canvas('canvas', {
		backgroundColor: '#ffffff',
		// width: 300,
		// height: 300,
		isDrawingMode: true
	});
	canvas.freeDrawingBrush.width = 5;
	canvas.freeDrawingBrush.color = '#000000';
	canvas.renderAll();
	canvas.on('mouse:up', predictSymbol);
	$('#clearBtn').on('click', clearCanvas)
	loadModel();
	loadSymbols();

};


const clearCanvas = () => {
	// canvas.remove(canvas.getObjects())
	canvas.clear();
	canvas.backgroundColor = '#ffffff';
	$('#symbol').html('');
	$('#probs').html('');
};

const loadModel = async () => {
	model = await tf.loadLayersModel('http://localhost:8080/model.json');
	console.log('Model loaded');
	document.getElementById('load').innerHTML = 'Model loaded.';

};

const loadSymbols = async () => {
	 await $.ajax({
		url: 'http://localhost:8080/symbols.csv',
		dataType:'text',
		success: function(data){
			let lst = data.split(/\r?\n|\r/)
			for(var i = 1 ; i < lst.length -1 ; i++){
					let row = lst[i].split(',')
					symbols[i-1] = {r:row[0], s:row[1]}
			}
		}
	});
}

const preprocessData = (imageData) => {
	return tf.tidy(() => {
		let imageTensor = tf.browser.fromPixels(imageData,1);
		let resizedImage = tf.image.resizeNearestNeighbor(imageTensor, [32, 32]).toFloat();
		// let resizedImage = tf.image.resizeBilinear(imageTensor, [ 32, 32 ]).toFloat();
		// resizedImage = resizedImage.slice([0, 0, 1], [32, 32, 1])

		// normalize image
		const normConst = tf.scalar(255.0);
		let normalizedImage = resizedImage.div(normConst);

		// Add a dimension to be suitable for prediction
		normalizedImage = normalizedImage.expandDims(0);
		return normalizedImage;
	});
};

const predictSymbol = () => {
	let testData = getImage();
	let probabilities = model.predict(testData).dataSync();
	let topK = findIndicesOfTopK(probabilities, 10);
	let topKProb = findTopKProb(probabilities, topK);
	let latexSymbols = findSymbols(topK)
	convert(latexSymbols)
	showProbs(topKProb)
};

const showProbs = topKProb => {
	$('#probs').html('')
	for (let index = 0; index < topKProb.length; index++) {
		const element = topKProb[index].toFixed(4);
		$('#probs').append('<p>'+element +'</p>')

	}

}

const convert = latexSymbols => {
	MathJax.texReset();
	var output = $('#symbol')[0]
	output.innerHTML = ''
	for (let index = 0; index < latexSymbols.length; index++) {
		const element = latexSymbols[index].r
		MathJax.tex2chtmlPromise(element).then(function (node) {
			output.appendChild(node);
			MathJax.startup.document.clear();
			MathJax.startup.document.updateDocument();
		}).catch(function (err) {
			console.log(err.message);
		})
}
}

const findSymbols = topK => {
	let latex = []
	for (let index = 0; index < topK.length; index++) {
		const element = topK[index];
		latex.push(symbols[element])
	}
	return latex
}

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
	let ctx = canvas.getContext('2d');
	const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
	return preprocessData(imageData);
};

window.onload = loadPage;
