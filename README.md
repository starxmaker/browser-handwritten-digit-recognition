# Browser Handwritten Number Recognition

A small library powered by MNIST ONNX model which runs an digit recognizer in the browser.

## Install

```bash
npm install browser-handwritten-number-recognition
```

## Usage

```html
<canvas id="draw" width="280" height="280"></canvas>
```

```javascript
import { recognizeDigit } from 'browser-handwritten-number-recognition'
const canvas = document.getElementById('draw')
const prediction = await recognizeDigit(canvas)
if (prediction) {
    console.log(`Digit: ${prediction.digit}`)
    console.log(`Confidence: ${(prediction.confidence * 100).toFixed(1)}%`)
} else {
    console.log("No digit recognized")
}
```

## Downloading model

```bash
npm run download-model
```

## Regenerating weights

If you modify or want to regenerate `weights.js` from the ONNX model, run:

```bash
npm run extract-weights
```