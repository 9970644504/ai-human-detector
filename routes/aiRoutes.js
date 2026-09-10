const express = require('express');
const router = express.Router();
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const sharp = require('sharp');
const ort = require('onnxruntime-node');
const getDb = require('../config/db');

const upload = multer({ 
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 } 
});

let detectSession = null;
let recogSession = null;

async function initModels() {
  const sessionOptions = {
    executionProviders: ['cpu'],
    graphOptimizationLevel: 'all',
    logSeverityLevel: 3
  };

  const detectPath = path.join(__dirname, '../models/version-RFB-320.onnx');
  const recogPath = path.join(__dirname, '../models/face_recognition.onnx');

  if (!detectSession && fs.existsSync(detectPath)) {
    detectSession = await ort.InferenceSession.create(detectPath, sessionOptions);
  }
  if (!recogSession && fs.existsSync(recogPath)) {
    recogSession = await ort.InferenceSession.create(recogPath, sessionOptions);
  }
}
initModels();

function l2Normalize(vec) {
  let sum = 0;
  for (let i = 0; i < vec.length; i++) sum += vec[i] * vec[i];
  const norm = Math.sqrt(sum);
  if (norm === 0) return vec;
  const res = new Float32Array(vec.length);
  for (let i = 0; i < vec.length; i++) res[i] = vec[i] / norm;
  return res;
}

function cosineSimilarity(vecA, vecB) {
  let dot = 0;
  for (let i = 0; i < vecA.length; i++) {
    dot += vecA[i] * vecB[i];
  }
  return dot;
}

// Extracts face embedding matching NHWC [1, 112, 112, 3] layout
async function extractSelfieEmbedding(imageBuffer) {
  if (!detectSession || !recogSession) await initModels();

  const image = sharp(imageBuffer).rotate();
  const meta = await image.metadata();
  const origW = meta.width;
  const origH = meta.height;

  if (!origW || !origH) return null;

  // 1. UltraFace Detection (320x240 NCHW)
  const { data: rgbData } = await image
    .clone()
    .resize(320, 240, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });

  const inputFloat32 = new Float32Array(1 * 3 * 240 * 320);
  const channelSize = 240 * 320;
  for (let i = 0; i < channelSize; i++) {
    inputFloat32[0 * channelSize + i] = (rgbData[i * 3 + 0] - 127.0) / 128.0;
    inputFloat32[1 * channelSize + i] = (rgbData[i * 3 + 1] - 127.0) / 128.0;
    inputFloat32[2 * channelSize + i] = (rgbData[i * 3 + 2] - 127.0) / 128.0;
  }

  const inputName = detectSession.inputNames[0];
  const results = await detectSession.run({
    [inputName]: new ort.Tensor('float32', inputFloat32, [1, 3, 240, 320])
  });

  const outValues = Object.values(results);
  const scores = outValues[0].data;
  const boxes = outValues[1].data;

  let bestBox = null;
  let largestArea = 0;

  const numBoxes = boxes.length / 4;
  for (let i = 0; i < numBoxes; i++) {
    const conf = scores[i * 2 + 1];
    if (conf > 0.30) {
      const w = boxes[i * 4 + 2] - boxes[i * 4 + 0];
      const h = boxes[i * 4 + 3] - boxes[i * 4 + 1];
      const area = w * h;

      if (area > largestArea) {
        largestArea = area;
        bestBox = [
          boxes[i * 4 + 0],
          boxes[i * 4 + 1],
          boxes[i * 4 + 2],
          boxes[i * 4 + 3]
        ];
      }
    }
  }

  if (!bestBox) return null;

  const rawX1 = Math.max(0, Math.min(1, bestBox[0]));
  const rawY1 = Math.max(0, Math.min(1, bestBox[1]));
  const rawX2 = Math.max(0, Math.min(1, bestBox[2]));
  const rawY2 = Math.max(0, Math.min(1, bestBox[3]));

  let rawLeft = Math.floor(rawX1 * origW);
  let rawTop = Math.floor(rawY1 * origH);
  let rawWidth = Math.ceil((rawX2 - rawX1) * origW);
  let rawHeight = Math.ceil((rawY2 - rawY1) * origH);

  // Square crop preserving aspect ratio
  const centerX = rawLeft + Math.floor(rawWidth / 2);
  const centerY = rawTop + Math.floor(rawHeight / 2);
  const maxSide = Math.max(rawWidth, rawHeight);
  const halfSide = Math.floor((maxSide * 1.25) / 2);

  let left = Math.max(0, centerX - halfSide);
  let top = Math.max(0, centerY - halfSide);
  let side = halfSide * 2;

  if (left + side > origW) side = origW - left;
  if (top + side > origH) side = origH - top;

  if (side <= 16) return null;

  // 2. Crop Face & Format to NHWC [1, 112, 112, 3]
  const faceBuffer = await sharp(imageBuffer)
    .rotate()
    .extract({ left, top, width: side, height: side })
    .resize(112, 112, { fit: 'fill' })
    .removeAlpha()
    .raw()
    .toBuffer();

  const totalPixels = 112 * 112 * 3;
  const recogFloatHWC = new Float32Array(totalPixels);
  for (let k = 0; k < totalPixels; k++) {
    recogFloatHWC[k] = (faceBuffer[k] - 127.5) / 128.0;
  }

  const recogInputName = recogSession.inputNames[0];
  const recogTensor = new ort.Tensor('float32', recogFloatHWC, [1, 112, 112, 3]);
  const recogOutput = await recogSession.run({ [recogInputName]: recogTensor });

  const rawVector = Object.values(recogOutput)[0].data;
  return l2Normalize(rawVector);
}

// POST /api/ai/match — Event-wide Facial Search
router.post('/match', upload.single('selfie'), async (req, res) => {
  const startTime = Date.now();
  try {
    const { galleryId, eventId, slug } = req.body;
    const targetIdentifier = galleryId || eventId || slug;

    if (!req.file) {
      return res.status(400).json({ success: false, message: 'Selfie image is required.' });
    }

    console.log(`\n======================================================`);
    console.log(`[AI Match] Processing search request (${req.file.size} bytes)...`);

    const selfieVector = await extractSelfieEmbedding(req.file.buffer);
    if (!selfieVector) {
      return res.status(400).json({ 
        success: false, 
        message: 'Could not detect a clear face in your selfie. Please face the camera directly in good lighting.' 
      });
    }

    const db = await getDb();
    let query = 'SELECT id, event_id, filename, url, face_embeddings FROM photos';
    let params = [];

    if (targetIdentifier && targetIdentifier !== 'all') {
      const eventRecord = await db.get(
        'SELECT id FROM events WHERE id = ? OR slug = ?', 
        [targetIdentifier, targetIdentifier]
      );
      if (eventRecord) {
        query += ' WHERE event_id = ?';
        params.push(eventRecord.id);
      }
    }

    const photos = await db.all(query, params);
    console.log(`[AI Match] Searching across ${photos.length} total event photos...`);

    // Operational threshold for real-world variation
    const MATCH_THRESHOLD = 0.36;
    const matchedPhotos = [];
    const allPhotoScores = [];

    for (const photo of photos) {
      let embeddingsList = [];
      try {
        embeddingsList = JSON.parse(photo.face_embeddings || '[]');
      } catch (e) {
        embeddingsList = [];
      }

      if (!embeddingsList.length) continue;

      let highestScoreForThisPhoto = 0;

      for (const embArr of embeddingsList) {
        const storedVector = new Float32Array(embArr);
        const similarity = cosineSimilarity(selfieVector, storedVector);
        if (similarity > highestScoreForThisPhoto) {
          highestScoreForThisPhoto = similarity;
        }
      }

      allPhotoScores.push({ filename: photo.filename, score: highestScoreForThisPhoto });

      if (highestScoreForThisPhoto >= MATCH_THRESHOLD) {
        let photoUrl = photo.url || `/uploads/galleries/${photo.event_id}/${photo.filename}`;
        photoUrl = photoUrl.replace(/\\/g, '/');

        matchedPhotos.push({
          id: photo.id,
          filename: photo.filename,
          url: photoUrl,
          similarity: Number(highestScoreForThisPhoto.toFixed(4))
        });
      }
    }

    matchedPhotos.sort((a, b) => b.similarity - a.similarity);
    allPhotoScores.sort((a, b) => b.score - a.score);

    console.log(`\n--- GALLERY SEARCH RESULTS ---`);
    allPhotoScores.slice(0, 10).forEach((item, idx) => {
      console.log(`  #${idx + 1}: ${item.filename} => Score: ${item.score.toFixed(3)} ${item.score >= MATCH_THRESHOLD ? '✅ [MATCH]' : '❌ [BELOW]'}`);
    });

    const timeTaken = Date.now() - startTime;
    console.log(`\n[AI Match] Search complete in ${timeTaken}ms! Found ${matchedPhotos.length} match(es).`);
    console.log(`======================================================\n`);

    return res.status(200).json({
      success: true,
      total_matches: matchedPhotos.length,
      time_taken_ms: timeTaken,
      matches: matchedPhotos
    });

  } catch (err) {
    console.error('[AI Match Error Details]:', err);
    return res.status(500).json({ 
      success: false, 
      message: err.message || 'Error running face search on server.' 
    });
  }
});

module.exports = router;