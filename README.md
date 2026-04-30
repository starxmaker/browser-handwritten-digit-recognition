# Browser Handwritten Digit Recognition

A small library powered by MNIST ONNX model which runs an digit recognizer in the browser.

You can check a live demo [here](https://starxmaker.github.io/browser-handwritten-digit-recognition/)

## Install

```bash
npm install browser-handwritten-digit-recognition
```

## Usage

Given a HTML canva element:

```html
<canvas id="draw" width="280" height="280"></canvas>
```
After drawing something on it, you recognize the digit with:

```javascript
import { recognizeDigit } from 'browser-handwritten-digit-recognition'
const canvas = document.getElementById('draw')
const prediction = await recognizeDigit(canvas)
if (prediction) {
    console.log(`Digit: ${prediction.digit}`)
    console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`)
} else {
    console.log("No digit recognized")
}
```

## Generate required source files

This repo does not include the MNIST-8 model, to download it use:

```bash
npm run download-model
```

Then, it is required to generate a `weights.js` file that can be used by the library.

```bash
npm run extract-weights
```

## Attributions
- [Mnist - Handwritten digit recognition](https://github.com/onnx/models/tree/main/validated/vision/classification/mnist)