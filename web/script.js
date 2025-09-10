/* global ort */
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

// Set black background and white stroke, to match training data
ctx.fillStyle = 'black';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = 'white';
ctx.lineWidth = 22;

let drawing = false;
let lastX = 0, lastY = 0;

canvas.addEventListener('mousedown', (e) => {
  drawing = true;
  [lastX, lastY] = [e.offsetX, e.offsetY];
});
canvas.addEventListener('mousemove', (e) => {
  if (!drawing) return;
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(e.offsetX, e.offsetY);
  ctx.stroke();
  [lastX, lastY] = [e.offsetX, e.offsetY];
});
canvas.addEventListener('mouseup', () => drawing = false);
canvas.addEventListener('mouseleave', () => drawing = false);

// Touch support
canvas.addEventListener('touchstart', (e) => {
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const t = e.touches[0];
  drawing = true;
  [lastX, lastY] = [t.clientX - rect.left, t.clientY - rect.top];
}, { passive: false });

canvas.addEventListener('touchmove', (e) => {
  e.preventDefault();
  if (!drawing) return;
  const rect = canvas.getBoundingClientRect();
  const t = e.touches[0];
  const x = t.clientX - rect.left;
  const y = t.clientY - rect.top;
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(x, y);
  ctx.stroke();
  [lastX, lastY] = [x, y];
}, { passive: false });

canvas.addEventListener('touchend', () => drawing = false);

document.getElementById('clearBtn').addEventListener('click', () => {
  ctx.fillStyle = 'black';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = 'white';
  document.getElementById('pred').textContent = '-';
  document.getElementById('conf').textContent = '-';
});

document.getElementById('saveOBtn').addEventListener('click', () => saveSample('O'));
document.getElementById('saveXBtn').addEventListener('click', () => saveSample('X'));

function saveSample(label) {
  const link = document.createElement('a');
  link.download = `${label}_${Date.now()}.png`;
  link.href = canvas.toDataURL('image/png');
  link.click();
}

function preprocessTo28x28() {
  // Downscale to 28x28 on an offscreen canvas
  const off = document.createElement('canvas');
  off.width = 28; off.height = 28;
  const octx = off.getContext('2d');
  octx.drawImage(canvas, 0, 0, 28, 28);
  const { data } = octx.getImageData(0, 0, 28, 28);
  // data is RGBA; use R as grayscale since we draw white on black
  const arr = new Float32Array(1 * 1 * 28 * 28);
  for (let i = 0; i < 28 * 28; i++) {
    const r = data[i * 4];
    arr[i] = r / 255.0; // 0..1
  }
  return arr;
}

async function predict() {
  const statusEl = document.getElementById('status');
  try {
    statusEl.textContent = '加载模型中...';
    const session = await ort.InferenceSession.create('model/model.onnx', {
      executionProviders: ['wasm']
    });
    const inputData = preprocessTo28x28();
    const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
    statusEl.textContent = '推理中...';
    const outputs = await session.run({ input: tensor });
    const logits = outputs.output.data;
    // softmax
    const maxLogit = Math.max(logits[0], logits[1]);
    const exps = [Math.exp(logits[0] - maxLogit), Math.exp(logits[1] - maxLogit)];
    const sum = exps[0] + exps[1];
    const probs = [exps[0] / sum, exps[1] / sum];
    const labels = ['O', 'X'];
    let predIdx = probs[0] >= probs[1] ? 0 : 1;
    document.getElementById('pred').textContent = labels[predIdx];
    document.getElementById('conf').textContent = (probs[predIdx] * 100).toFixed(2) + '%';
    statusEl.textContent = '完成';
  } catch (e) {
    console.error(e);
    statusEl.textContent = '模型加载或推理失败。请确保 model/model.onnx 存在。';
  }
}

document.getElementById('predictBtn').addEventListener('click', predict);