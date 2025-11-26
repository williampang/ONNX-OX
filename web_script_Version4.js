/* global ort */
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const CLASSES = ['O', 'V', 'X'];
const MODEL_CANDIDATES = ['model/model_v4_handwritten.onnx', 'model/model.onnx'];
let sessionPromise = null;
let activeModelPath = null;

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
  renderProbabilities(null);
});

document.getElementById('saveOBtn').addEventListener('click', () => saveSample('O'));
document.getElementById('saveVBtn').addEventListener('click', () => saveSample('V'));
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
    const session = await getSession(statusEl);
    const inputData = preprocessTo28x28();
    const tensor = new ort.Tensor('float32', inputData, [1, 1, 28, 28]);
    statusEl.textContent = '推理中...';
    const outputs = await session.run({ input: tensor });
    const logits = Array.from(outputs.output.data);
    if (logits.length !== CLASSES.length) {
      throw new Error(`模型输出维度 ${logits.length} 与预期 ${CLASSES.length} 不符`);
    }
    const probs = softmax(logits);
    const predIdx = probs.indexOf(Math.max(...probs));
    document.getElementById('pred').textContent = CLASSES[predIdx];
    document.getElementById('conf').textContent = (probs[predIdx] * 100).toFixed(2) + '%';
    renderProbabilities(probs);
    statusEl.textContent = `完成：${activeModelPath || '未知模型'}`;
  } catch (e) {
    console.error(e);
    sessionPromise = null;
    statusEl.textContent = '模型加载或推理失败，请检查模型文件路径。';
  }
}

document.getElementById('predictBtn').addEventListener('click', predict);

async function getSession(statusEl) {
  if (sessionPromise) return sessionPromise;
  sessionPromise = (async () => {
    let lastError = null;
    for (const path of MODEL_CANDIDATES) {
      try {
        statusEl.textContent = `加载模型：${path}`;
        const session = await ort.InferenceSession.create(path, {
          executionProviders: ['wasm']
        });
        activeModelPath = path;
        return session;
      } catch (err) {
        console.warn(`无法加载 ${path}`, err);
        lastError = err;
      }
    }
    throw lastError || new Error('未找到可用的模型文件');
  })();
  return sessionPromise;
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((v) => v / sum);
}

function renderProbabilities(probs) {
  const tbody = document.getElementById('probBody');
  tbody.innerHTML = '';
  if (!probs) return;
  const maxIdx = probs.indexOf(Math.max(...probs));
  probs.forEach((p, idx) => {
    const tr = document.createElement('tr');
    if (idx === maxIdx) tr.classList.add('highlight');
    const tdLabel = document.createElement('td');
    tdLabel.textContent = CLASSES[idx];
    const tdProb = document.createElement('td');
    tdProb.textContent = (p * 100).toFixed(2) + '%';
    tr.appendChild(tdLabel);
    tr.appendChild(tdProb);
    tbody.appendChild(tr);
  });
}