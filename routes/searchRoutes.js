const express = require('express');
const multer = require('multer');
const path = require('path');
const fs = require('fs');
const { spawn } = require('child_process');
const db = require('../services/db');
const { searchMatchingFaces } = require('../services/vectorDbService');

const router = express.Router();
const upload = multer({ dest: path.join(__dirname, '../uploads/temp/') });

// Cosine similarity helper for SQLite fallback matching
function cosineSimilarity(vecA, vecB) {
  let dotProduct = 0.0;
  let normA = 0.0;
  let normB = 0.0;
  for (let i = 0; i < vecA.length; i++) {
    dotProduct += vecA[i] * vecB[i];
    normA += vecA[i] * vecA[i];
    normB += vecB[i] * vecB[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
}

router.post('/:slug', upload.single('selfie'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'Selfie image is required.' });
  }

  const selfiePath = req.file.path;
  const slug = req.params.slug;

  const gallery = db.prepare('SELECT * FROM galleries WHERE slug = ?').get(slug);
  if (!gallery) {
    if (fs.existsSync(selfiePath)) fs.unlinkSync(selfiePath);
    return res.status(404).json({ error: 'Gallery not found.' });
  }

  // 1. Extract vector only from the client's single uploaded selfie
  const pythonPath = path.join(__dirname, '../.venv/Scripts/python.exe');
  const scriptPath = path.join(__dirname, '../services/aiWorker.py');

  const pythonProcess = spawn(pythonPath, [scriptPath]);
  const payload = { action: 'extract_embedding', imagePath: selfiePath };

  let outputData = '';
  pythonProcess.stdin.write(JSON.stringify(payload));
  pythonProcess.stdin.end();

  pythonProcess.stdout.on('data', (data) => {
    outputData += data.toString();
  });

  pythonProcess.on('close', async () => {
    if (fs.existsSync(selfiePath)) fs.unlinkSync(selfiePath);

    try {
      const result = JSON.parse(outputData);
      if (result.status !== 'success' || !result.embedding) {
        return res.json({ status: 'error', message: 'No face detected in reference selfie.' });
      }

      const selfieVector = result.embedding;
      let matches = [];

      // 2. Try fast search in Qdrant Vector DB first
      const hits = await searchMatchingFaces(selfieVector, gallery.id, 50, 0.45);
      if (hits && hits.length > 0) {
        matches = hits.map(m => m.photoUrl);
      } else {
        // 3. Fallback: Fast vector matching in Node.js using SQLite cached embeddings
        const photos = db.prepare('SELECT photo_url, embedding FROM photos WHERE gallery_id = ? AND embedding IS NOT NULL').all(gallery.id);
        
        photos.forEach((photo) => {
          const cachedVector = JSON.parse(photo.embedding);
          const score = cosineSimilarity(selfieVector, cachedVector);
          if (score >= 0.45) {
            matches.push(photo.photo_url);
          }
        });

        // If no embedding matched yet, return all gallery photos for preview
        if (matches.length === 0) {
          const allPhotos = db.prepare('SELECT photo_url FROM photos WHERE gallery_id = ?').all(gallery.id);
          matches = allPhotos.map(p => p.photo_url);
        }
      }

      return res.json({ status: 'success', matches });
    } catch (err) {
      return res.status(500).json({ error: 'Failed to process face match.' });
    }
  });
});

module.exports = router;