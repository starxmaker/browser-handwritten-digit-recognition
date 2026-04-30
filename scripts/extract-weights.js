// scripts/extract-weights.js
// Extract MNIST-8 ONNX weights and emit a JS module with inlined Float32Arrays.
// scripts/extract-weights.js
const fs = require('fs');
const path = require('path');
const onnx = require('onnx-proto');

const modelPath = path.resolve('./mnist-8.onnx');
const outPath = path.resolve('./weights.js');

function formatFloat(x, precision = 6) {
  if (!Number.isFinite(x)) return String(x);
  if (x === 0) return '0';

  const abs = Math.abs(x);
  const exp = Math.floor(Math.log10(abs));

  let s;
  if (exp < -4 || exp >= precision) {
    s = x.toExponential(precision - 1);
    s = s.replace(/(\.\d*?[1-9])0+e/, '$1e');
    s = s.replace(/\.0+e/, 'e');
    s = s.replace(/e([+-])0+(\d+)/, 'e$1$2');
  } else {
    s = x.toPrecision(precision);
    s = s.replace(/(\.\d*?[1-9])0+$/, '$1');
    s = s.replace(/\.0+$/, '');
  }

  return s;
}

function fmt(floats) {
  return floats.map(f => formatFloat(f, 6)).join(',');
}

function getFloatsFromInitializer(init) {
  if (init.rawData && init.rawData.length > 0) {
    const buf = Buffer.from(init.rawData);
    const floats = [];
    for (let i = 0; i < buf.length; i += 4) {
      floats.push(buf.readFloatLE(i));
    }
    return floats;
  }

  if (init.floatData && init.floatData.length > 0) {
    return Array.from(init.floatData);
  }

  return [];
}

async function main() {
  const data = fs.readFileSync(modelPath);
  const model = onnx.onnx.ModelProto.decode(data);
  const graph = model.graph;

  const weights = {};

  for (const init of graph.initializer) {
    const floats = getFloatsFromInitializer(init);
    if (floats.length > 0) {
      weights[init.name] = floats;
    }
  }

  let output = '';
  output += '// AUTO-GENERATED FILE.\n';
  output += '// Derived from `mnist-8.onnx` from the ONNX Model Zoo MNIST model.\n';
  output += '// Upstream model license: MIT.\n';
  output += '// See LICENSE and THIRD_PARTY_NOTICES.md.\n';
  output += '// MNIST-8 model weights\n\n';

  for (const [name, floats] of Object.entries(weights)) {
    const safe = name.replace(/[/.]/g, '_');
    output += `export const ${safe} = new Float32Array([${fmt(floats)}]);\n`;
  }

  fs.mkdirSync(path.dirname(outPath), { recursive: true });
  fs.writeFileSync(outPath, output, 'utf8');

  const size = fs.statSync(outPath).size;
  console.log('Generated', outPath, ':', size, 'bytes');
  for (const [name, floats] of Object.entries(weights)) {
    console.log(' ', name, ':', floats.length, 'floats');
  }
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});
