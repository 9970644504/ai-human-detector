const path = require('path');
const fs = require('fs');
const sharp = require('sharp');
const ort = require('onnxruntime-node');

function dot(a, b) {
  return a.reduce((sum, val, i) => sum + val * b[i], 0);
}

function l2Norm(vec) {
  let sum = 0.0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  const mag = Math.sqrt(sum);
  return mag === 0 ? vec : vec.map(x => x / mag);
}

function softmax(val0, val1) {
  const max = Math.max(val0, val1);
  const exp0 = Math.exp(val0 - max);
  const exp1 = Math.exp(val1 - max);
  return exp1 / (exp0 + exp1);
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

async function detectFacesStandalone(filePath, detSession) {
  const meta = await sharp(filePath).rotate().metadata();
  const origW = meta.width;
  const origH = meta.height;

  const rawData = await sharp(filePath)
    .rotate()
    .resize(320, 240, { fit: 'contain', background: { r: 127, g: 127, b: 127 } })
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
    if (score > 0.45) {
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

async function testAll() {
  const detSession = await ort.InferenceSession.create(
    path.join(__dirname, 'models/version-RFB-320.onnx'),
    { executionProviders: ['cpu'], logSeverityLevel: 3 }
  );
  const recSession = await ort.InferenceSession.create(
    path.join(__dirname, 'models/face_recognition.onnx'),
    { executionProviders: ['cpu'], logSeverityLevel: 3 }
  );

  const debugDir = path.join(__dirname, 'debug_crops');
  if (!fs.existsSync(debugDir)) fs.mkdirSync(debugDir);

  const imgA = 'C:\\Users\\dell\\OneDrive\\Desktop\\ai-human-detector\\uploads\\galleries\\hello\\Rohit_Sharma\\1788777058995-183268.jpg';
  const imgB = 'C:\\Users\\dell\\OneDrive\\Desktop\\ai-human-detector\\uploads\\galleries\\hello\\Rohit_Sharma\\1788777059003-499973.jpg';

  async function extractFeatures(filePath, label, mode) {
    const boxes = await detectFacesStandalone(filePath, detSession);
    const embs = [];

    for (let idx = 0; idx < boxes.length; idx++) {
      const [x1, y1, x2, y2] = boxes[idx];
      const w = Math.max(1, x2 - x1);
      const h = Math.max(1, y2 - y1);
      const cx = x1 + w / 2;
      const cy = y1 + h / 2;
      const side = Math.max(w, h) * 1.30;

      const meta = await sharp(filePath).rotate().metadata();
      const left = Math.max(0, Math.round(cx - side / 2));
      const top = Math.max(0, Math.round(cy - side / 2));
      const extractW = Math.min(meta.width - left, Math.round(side));
      const extractH = Math.min(meta.height - top, Math.round(side));

      const raw = await sharp(filePath)
        .rotate()
        .extract({ left, top, width: extractW, height: extractH })
        .resize(112, 112, { fit: 'fill' })
        .removeAlpha()
        .raw()
        .toBuffer();

      // Save crop on disk once for visual confirmation
      if (mode === 'rgb') {
        await sharp(raw, { raw: { width: 112, height: 112, channels: 3 } })
          .toFile(path.join(debugDir, `${label}_face_${idx}.jpg`));
      }

      const input = new Float32Array(112 * 112 * 3);
      for (let i = 0; i < 112 * 112; i++) {
        const r = raw[i * 3 + 0];
        const g = raw[i * 3 + 1];
        const b = raw[i * 3 + 2];

        if (mode === 'rgb') {
          input[i * 3 + 0] = (r - 127.5) / 127.5;
          input[i * 3 + 1] = (g - 127.5) / 127.5;
          input[i * 3 + 2] = (b - 127.5) / 127.5;
        } else {
          input[i * 3 + 0] = (b - 127.5) / 127.5;
          input[i * 3 + 1] = (g - 127.5) / 127.5;
          input[i * 3 + 2] = (r - 127.5) / 127.5;
        }
      }

      const tensor = new ort.Tensor('float32', input, [1, 112, 112, 3]);
      const res = await recSession.run({ [recSession.inputNames[0]]: tensor });
      embs.push(l2Norm(Array.from(res[recSession.outputNames[0]].data)));
    }
    return embs;
  }

  console.log('--- TESTING RGB ORDER ---');
  const rgbA = await extractFeatures(imgA, 'A', 'rgb');
  const rgbB = await extractFeatures(imgB, 'B', 'rgb');

  for (let i = 0; i < rgbA.length; i++) {
    for (let j = 0; j < rgbB.length; j++) {
      const sim = dot(rgbA[i], rgbB[j]);
      console.log(`RGB: Face A[${i}] vs Face B[${j}] -> Score: ${sim.toFixed(4)}`);
    }
  }

  console.log('\n--- TESTING BGR ORDER ---');
  const bgrA = await extractFeatures(imgA, 'A', 'bgr');
  const bgrB = await extractFeatures(imgB, 'B', 'bgr');

  for (let i = 0; i < bgrA.length; i++) {
    for (let j = 0; j < bgrB.length; j++) {
      const sim = dot(bgrA[i], bgrB[j]);
      console.log(`BGR: Face A[${i}] vs Face B[${j}] -> Score: ${sim.toFixed(4)}`);
    }
  }

  console.log(`\nCrop images saved into: ${debugDir}`);
  process.exit(0);
}

testAll().catch(console.error);