const ort = require('onnxruntime-node');
const sharp = require('sharp');
const path = require('path');
const fs = require('fs');
const getDb = require('../config/db');

let detSession = null;
let recSession = null;

const DET_MODEL_PATH = path.join(__dirname, '../models/version-RFB-320.onnx');
const REC_MODEL_PATH = path.join(__dirname, '../models/face_recognition.onnx');

function l2Norm(vec) {
  let sum = 0.0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  const mag = Math.sqrt(sum);
  if (mag === 0) return vec;
  return vec.map(x => x / mag);
}

function generatePriors() {
  const featureMaps = [[40, 30], [20, 15], [10, 8], [5, 4]];
  const minSizes = [[10, 16, 24], [32, 48], [64, 96], [128, 192, 256]];
  const steps = [8, 16, 32, 64];
  const priors = [];

  for (let k = 0; k < featureMaps.length; k++) {
    const [w, h] = featureMaps[k];
    const step = steps[k];
    const sizes = minSizes[k];

    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        const cx = (j + 0.5) * step / 320;
        const cy = (i + 0.5) * step / 240;
        for (const size of sizes) {
          priors.push([cx, cy, size / 320, size / 240]);
        }
      }
    }
  }
  return priors;
}

const PRIORS = generatePriors();

async function initModels() {
  if (detSession && recSession) return;

  const options = {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3
  };

  if (!fs.existsSync(DET_MODEL_PATH) || !fs.existsSync(REC_MODEL_PATH)) {
    throw new Error(`Models missing! Ensure version-RFB-320.onnx and face_recognition.onnx exist in /models.`);
  }

  detSession = await ort.InferenceSession.create(DET_MODEL_PATH, options);
  recSession = await ort.InferenceSession.create(REC_MODEL_PATH, options);
  console.log('[AI Engine] Loaded version-RFB-320 Detector and ArcFace Recognition Model.');
}

function softmax(val0, val1) {
  const max = Math.max(val0, val1);
  const exp0 = Math.exp(val0 - max);
  const exp1 = Math.exp(val1 - max);
  return exp1 / (exp0 + exp1);
}

async function detectFaces(filePath) {
  if (!detSession) await initModels();

  const meta = await sharp(filePath).rotate().metadata();
  const origW = meta.width;
  const origH = meta.height;

  const rawData = await sharp(filePath)
    .rotate()
    .resize(320, 240, {
      fit: 'contain',
      background: { r: 127, g: 127, b: 127 }
    })
    .removeAlpha()
    .raw()
    .toBuffer();

  const floatData = new Float32Array(3 * 320 * 240);
  for (let i = 0; i < 320 * 240; i++) {
    floatData[i] = (rawData[i * 3 + 0] - 127.0) / 128.0;
    floatData[320 * 240 + i] = (rawData[i * 3 + 1] - 127.0) / 128.0;
    floatData[2 * 320 * 240 + i] = (rawData[i * 3 + 2] - 127.0) / 128.0;
  }

  const tensor = new ort.Tensor('float32', floatData, [1, 3, 240, 320]);
  const feeds = { [detSession.inputNames[0]]: tensor };
  const results = await detSession.run(feeds);

  const scores = results[detSession.outputNames[0]].data;
  const boxes = results[detSession.outputNames[1]].data;

  const scale = Math.min(320 / origW, 240 / origH);
  const padX = (320 - origW * scale) / 2;
  const padY = (240 - origH * scale) / 2;

  const bboxes = [];
  for (let i = 0; i < PRIORS.length; i++) {
    const score = softmax(scores[i * 2], scores[i * 2 + 1]);
    if (score > 0.40) {
      const prior = PRIORS[i];
      const cx = (prior[0] + boxes[i * 4 + 0] * 0.1 * prior[2]) * 320;
      const cy = (prior[1] + boxes[i * 4 + 1] * 0.1 * prior[3]) * 240;
      const w = prior[2] * Math.exp(boxes[i * 4 + 2] * 0.2) * 320;
      const h = prior[3] * Math.exp(boxes[i * 4 + 3] * 0.2) * 240;

      let x1 = (cx - w / 2 - padX) / scale;
      let y1 = (cy - h / 2 - padY) / scale;
      let x2 = (cx + w / 2 - padX) / scale;
      let y2 = (cy + h / 2 - padY) / scale;

      x1 = Math.max(0, Math.round(x1));
      y1 = Math.max(0, Math.round(y1));
      x2 = Math.min(origW, Math.round(x2));
      y2 = Math.min(origH, Math.round(y2));

      if (x2 - x1 > 16 && y2 - y1 > 16) {
        bboxes.push([x1, y1, x2, y2, score]);
      }
    }
  }

  bboxes.sort((a, b) => b[4] - a[4]);
  const finalBoxes = [];
  for (const box of bboxes) {
    let keep = true;
    for (const chosen of finalBoxes) {
      const interX1 = Math.max(box[0], chosen[0]);
      const interY1 = Math.max(box[1], chosen[1]);
      const interX2 = Math.min(box[2], chosen[2]);
      const interY2 = Math.min(box[3], chosen[3]);

      const interW = Math.max(0, interX2 - interX1);
      const interH = Math.max(0, interY2 - interY1);
      const interArea = interW * interH;

      const areaA = (box[2] - box[0]) * (box[3] - box[1]);
      const areaB = (chosen[2] - chosen[0]) * (chosen[3] - chosen[1]);
      const iou = interArea / (areaA + areaB - interArea);

      if (iou > 0.40) {
        keep = false;
        break;
      }
    }
    if (keep) finalBoxes.push(box);
  }

  return finalBoxes;
}

// Tight ArcFace Crop with BGR Ordering
async function extractFaceEmbedding(filePath, bbox) {
  if (!recSession) await initModels();

  const [x1, y1, x2, y2] = bbox;
  const width = Math.max(1, x2 - x1);
  const height = Math.max(1, y2 - y1);

  // Standard ArcFace crop: 1.08x box ratio
  const cx = x1 + width / 2;
  const cy = y1 + height / 2;
  const side = Math.max(width, height) * 1.08;

  const meta = await sharp(filePath).rotate().metadata();
  const left = Math.max(0, Math.round(cx - side / 2));
  const top = Math.max(0, Math.round(cy - side / 2));
  const extractW = Math.min(meta.width - left, Math.round(side));
  const extractH = Math.min(meta.height - top, Math.round(side));

  const rawCrop = await sharp(filePath)
    .rotate()
    .extract({ left, top, width: extractW, height: extractH })
    .resize(112, 112, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer();

  // BGR Format: NHWC [1, 112, 112, 3]
  const inputFloats = new Float32Array(112 * 112 * 3);
  for (let i = 0; i < 112 * 112; i++) {
    const r = rawCrop[i * 3 + 0];
    const g = rawCrop[i * 3 + 1];
    const b = rawCrop[i * 3 + 2];

    inputFloats[i * 3 + 0] = (b - 127.5) / 127.5; // Blue
    inputFloats[i * 3 + 1] = (g - 127.5) / 127.5; // Green
    inputFloats[i * 3 + 2] = (r - 127.5) / 127.5; // Red
  }

  const tensor = new ort.Tensor('float32', inputFloats, [1, 112, 112, 3]);
  const feeds = { [recSession.inputNames[0]]: tensor };
  const results = await recSession.run(feeds);
  const rawEmbeddings = results[recSession.outputNames[0]].data;

  return l2Norm(Array.from(rawEmbeddings));
}

async function getEmbeddingsFromFile(filePath) {
  let boxes = await detectFaces(filePath);

  if (boxes.length === 0) {
    const meta = await sharp(filePath).rotate().metadata();
    const w = meta.width;
    const h = meta.height;
    boxes = [[Math.round(w * 0.1), Math.round(h * 0.1), Math.round(w * 0.9), Math.round(h * 0.9), 0.99]];
  }

  const embeddings = [];
  for (const box of boxes) {
    try {
      const emb = await extractFaceEmbedding(filePath, box);
      embeddings.push(emb);
    } catch (err) {
      console.warn('[Embed Skip]:', err.message);
    }
  }

  return embeddings;
}

async function indexPhotoFaces(photoId, filePath) {
  try {
    const embeddings = await getEmbeddingsFromFile(filePath);
    const db = await getDb();
    await db.run(
      'UPDATE photos SET face_embeddings = ? WHERE id = ?',
      [JSON.stringify(embeddings), photoId]
    );
    return embeddings.length;
  } catch (err) {
    console.error(`[Index Error for ${photoId}]:`, err.message);
    return 0;
  }
}

module.exports = {
  initModels,
  detectFaces,
  getEmbeddingsFromFile,
  indexPhotoFaces,
  l2Norm
};