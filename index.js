/**
 * Pure-JavaScript MNIST digit recognizer (no WASM, no external runtime).
 * Implements the MNIST-8 CNN forward pass from the ONNX Model Zoo.
 * Architecture: Conv(8,5×5,same) → ReLU → MaxPool(2,2) →
 *               Conv(16,5×5,same) → ReLU → MaxPool(3,3) → FC(10)
 * Weights are lazily loaded from a dynamic import (bundled chunk, always offline).
 */

let weightsModule = null

async function getWeights() {
  if (!weightsModule) {
    weightsModule = await import('./weights.js')
  }
  return weightsModule
}

function relu(a) {
  for (let i = 0; i < a.length; i++) {
    if (a[i] < 0) a[i] = 0
  }
}

/** 2D convolution with SAME_UPPER padding (pad = floor(k/2) per side for odd kernels). */
function conv2dSame(input, inC, inH, inW, kernel, outC, kH, kW, bias) {
  const padH = (kH - 1) >> 1
  const padW = (kW - 1) >> 1
  const output = new Float32Array(outC * inH * inW)

  for (let oc = 0; oc < outC; oc++) {
    const b = bias[oc]
    const kOcBase = oc * inC * kH * kW
    for (let oh = 0; oh < inH; oh++) {
      for (let ow = 0; ow < inW; ow++) {
        let sum = b
        for (let ic = 0; ic < inC; ic++) {
          const kIcBase = kOcBase + ic * kH * kW
          const inIcBase = ic * inH * inW
          for (let kh = 0; kh < kH; kh++) {
            const ih = oh - padH + kh
            if (ih < 0 || ih >= inH) continue
            const kHBase = kIcBase + kh * kW
            const inHBase = inIcBase + ih * inW
            for (let kw = 0; kw < kW; kw++) {
              const iw = ow - padW + kw
              if (iw < 0 || iw >= inW) continue
              sum += input[inHBase + iw] * kernel[kHBase + kw]
            }
          }
        }
        output[(oc * inH + oh) * inW + ow] = sum
      }
    }
  }

  return output
}

/** 2D max pooling, VALID padding. */
function maxPool2d(input, inC, inH, inW, kH, kW, stride) {
  const outH = Math.floor((inH - kH) / stride) + 1
  const outW = Math.floor((inW - kW) / stride) + 1
  const out = new Float32Array(inC * outH * outW)

  for (let c = 0; c < inC; c++) {
    const cIn = c * inH * inW
    const cOut = c * outH * outW
    for (let oh = 0; oh < outH; oh++) {
      for (let ow = 0; ow < outW; ow++) {
        let maxVal = -Infinity
        for (let kh = 0; kh < kH; kh++) {
          const ih = oh * stride + kh
          const base = cIn + ih * inW
          for (let kw = 0; kw < kW; kw++) {
            const v = input[base + ow * stride + kw]
            if (v > maxVal) maxVal = v
          }
        }
        out[cOut + oh * outW + ow] = maxVal
      }
    }
  }

  return { out, outH, outW }
}

function softmax(arr) {
  let max = arr[0]
  for (let i = 1; i < arr.length; i++) {
    if (arr[i] > max) max = arr[i]
  }

  let sum = 0
  const out = new Float32Array(arr.length)
  for (let i = 0; i < arr.length; i++) {
    out[i] = Math.exp(arr[i] - max)
    sum += out[i]
  }
  for (let i = 0; i < out.length; i++) {
    out[i] /= sum
  }
  return out
}

/**
 * Preprocess a drawing canvas to a [784] Float32Array for MNIST inference.
 * - Auto-detects whether the canvas has a dark or light background
 * - Normalizes to white-bg / black-ink, crops to ink bounding box, pads, scales to 28×28
 * - MNIST convention: 1 = ink, 0 = background
 */
function preprocessCanvas(srcCanvas) {
  const ctx = srcCanvas.getContext('2d')
  const { width: W, height: H } = srcCanvas
  const data = ctx.getImageData(0, 0, W, H).data

  function cornerGray(x, y) {
    const i = (y * W + x) * 4
    return 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
  }

  const bgGray =
    (cornerGray(0, 0) +
      cornerGray(W - 1, 0) +
      cornerGray(0, H - 1) +
      cornerGray(W - 1, H - 1)) / 4

  const darkBg = bgGray < 128

  let minX = W
  let maxX = -1
  let minY = H
  let maxY = -1

  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = (y * W + x) * 4
      const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2]
      const isInk = darkBg ? gray > 128 : gray < 128
      if (isInk) {
        if (x < minX) minX = x
        if (x > maxX) maxX = x
        if (y < minY) minY = y
        if (y > maxY) maxY = y
      }
    }
  }

  if (maxX < minX || maxY < minY) return null

  let drawSrc = srcCanvas
  if (darkBg) {
    const inv = document.createElement('canvas')
    inv.width = W
    inv.height = H
    const ic = inv.getContext('2d')
    const invData = new Uint8ClampedArray(data.length)
    for (let i = 0; i < data.length; i++) {
      invData[i] = i % 4 === 3 ? data[i] : 255 - data[i]
    }
    ic.putImageData(new ImageData(invData, W, H), 0, 0)
    drawSrc = inv
  }

  const inkW = maxX - minX + 1
  const inkH = maxY - minY + 1
  const sq = Math.max(inkW, inkH)
  const pad = Math.ceil(sq * 0.25)
  const total = sq + 2 * pad
  const cx = (minX + maxX) / 2
  const cy = (minY + maxY) / 2
  const half = total / 2

  const tmp = document.createElement('canvas')
  tmp.width = 28
  tmp.height = 28
  const tc = tmp.getContext('2d')
  tc.fillStyle = 'white'
  tc.fillRect(0, 0, 28, 28)
  tc.drawImage(drawSrc, cx - half, cy - half, total, total, 0, 0, 28, 28)

  const px = tc.getImageData(0, 0, 28, 28).data
  const result = new Float32Array(784)
  for (let i = 0; i < 784; i++) {
    const r = px[i * 4]
    const g = px[i * 4 + 1]
    const b = px[i * 4 + 2]
    result[i] = 1 - (0.299 * r + 0.587 * g + 0.114 * b) / 255
  }

  return result
}

/**
 * Recognize a handwritten Sudoku digit (1-9) from a canvas.
 * Returns null if the canvas is blank or confidence is too low.
 * First call dynamically loads the weights (~60KB bundled chunk).
 */
export async function recognizeDigit(canvas) {
  const input = preprocessCanvas(canvas)
  if (!input) return null

  const w = await getWeights()

  let x = conv2dSame(input, 1, 28, 28, w.Parameter5, 8, 5, 5, w.Parameter6)
  relu(x)

  const p1 = maxPool2d(x, 8, 28, 28, 2, 2, 2)

  x = conv2dSame(p1.out, 8, p1.outH, p1.outW, w.Parameter87, 16, 5, 5, w.Parameter88)
  relu(x)

  const p2 = maxPool2d(x, 16, p1.outH, p1.outW, 3, 3, 3)

  const logits = new Float32Array(10)
  for (let j = 0; j < 10; j++) {
    let sum = w.Parameter194[j]
    for (let i = 0; i < 256; i++) {
      sum += p2.out[i] * w.Parameter193[i * 10 + j]
    }
    logits[j] = sum
  }

  const probs = softmax(logits)
  let best = 1
  for (let i = 2; i <= 9; i++) {
    if (probs[i] > probs[best]) best = i
  }

  const confidence = probs[best]
  if (confidence < 0.10) return null

  return { digit: best, confidence }
}
