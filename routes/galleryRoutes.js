const express = require('express');
const router = express.Router();
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const unzipper = require('unzipper');
const AdmZip = require('adm-zip');
const getDb = require('../config/db');
const { requireAuth } = require('../middleware/authMiddleware');

function cleanFolderName(name) {
  if (!name || !name.trim()) return 'General';
  return name
    .trim()
    .replace(/[<>:"/\\|?*\x00-\x1F]/g, '')
    .replace(/\s+/g, '_');
}

// --------------------------------------------------------------------------
// 0. GLOBAL IN-MEMORY VECTOR CACHE (Sub-millisecond matching)
// --------------------------------------------------------------------------
const eventVectorCache = new Map();

function invalidateEventCache(eventId) {
  if (eventId) {
    eventVectorCache.delete(String(eventId));
  }
}

async function getEventVectors(db, eventId, eventSlug) {
  const cacheKey = String(eventId);
  if (eventVectorCache.has(cacheKey)) {
    return eventVectorCache.get(cacheKey);
  }

  const rows = await db.all(`
    SELECT fe.photo_id, fe.embedding, p.file_path as url, p.filename, p.folder_name
    FROM face_embeddings fe
    JOIN photos p ON fe.photo_id = p.id
    WHERE p.event_id = ? OR fe.event_id = ? OR fe.event_id = ?
  `, [eventId, eventId, eventSlug]);

  const parsedRecords = [];
  for (let i = 0; i < rows.length; i++) {
    const r = rows[i];
    try {
      const rawVector = typeof r.embedding === 'string' ? JSON.parse(r.embedding) : r.embedding;
      parsedRecords.push({
        photo_id: r.photo_id,
        url: r.url,
        filename: r.filename,
        folder_name: r.folder_name,
        vector: new Float32Array(rawVector)
      });
    } catch (e) {}
  }

  eventVectorCache.set(cacheKey, parsedRecords);
  return parsedRecords;
}

// --------------------------------------------------------------------------
// HTTP BRIDGE TO PERSISTENT PYTHON AI DAEMON (Port 5001)
// --------------------------------------------------------------------------
async function callAiDaemon(endpoint, payload) {
  const url = `http://127.0.0.1:5001/${endpoint}`;
  try {
    const response = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    if (!response.ok) {
      const errBody = await response.json().catch(() => ({}));
      throw new Error(errBody.message || `AI Daemon HTTP ${response.status}`);
    }

    return await response.json();
  } catch (err) {
    throw new Error(`AI Daemon unreachable on port 5001: ${err.message}`);
  }
}

// --------------------------------------------------------------------------
// 1. MULTER CONFIGURATIONS
// --------------------------------------------------------------------------
const diskStorage = multer.diskStorage({
  destination: (req, file, cb) => {
    const eventFolderName = req.eventCleanName || 'Event_Photos';
    let subFolder = 'All_Photos';
    if (req.body && req.body.folderName) {
      subFolder = cleanFolderName(req.body.folderName);
    } else if (file.originalname && file.originalname.includes('/')) {
      const parts = file.originalname.split('/');
      subFolder = cleanFolderName(parts[0]);
    }

    const destinationPath = path.join(__dirname, '..', 'uploads', 'galleries', eventFolderName, subFolder);
    if (!fs.existsSync(destinationPath)) {
      fs.mkdirSync(destinationPath, { recursive: true });
    }

    req.savedSubFolder = subFolder;
    cb(null, destinationPath);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname).toLowerCase();
    const cleanBase = path.basename(file.originalname, ext).replace(/[^a-zA-Z0-9_-]/g, '_');
    cb(null, `${cleanBase}_${Date.now()}${ext}`);
  }
});

const uploadPhotosMiddleware = multer({
  storage: diskStorage,
  limits: { fileSize: 100 * 1024 * 1024 }
}).array('photos', 300);

const zipUpload = multer({
  dest: path.join(__dirname, '..', 'uploads', 'temp_zips')
}).single('zipFile');

const selfieUpload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => {
      const tempDir = path.join(__dirname, '..', 'uploads', 'temp_selfies');
      if (!fs.existsSync(tempDir)) {
        fs.mkdirSync(tempDir, { recursive: true });
      }
      cb(null, tempDir);
    },
    filename: (req, file, cb) => {
      const ext = path.extname(file.originalname).toLowerCase() || '.jpg';
      cb(null, `selfie_${Date.now()}${ext}`);
    }
  }),
  limits: { fileSize: 15 * 1024 * 1024 }
}).single('selfie');

// -------------------------------------------------------------
// 2. GET /api/galleries — Photographer's Events
// -------------------------------------------------------------
router.get('/', requireAuth, async (req, res) => {
  try {
    const userId = req.user.id || req.user.userId;
    const db = await getDb();
    const events = await db.all(
      `SELECT id, name, slug, photo_count, created_at 
       FROM events 
       WHERE user_id = ? 
       ORDER BY created_at DESC`,
      [userId]
    );
    return res.status(200).json(events || []);
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Failed to retrieve events.' });
  }
});

// -------------------------------------------------------------
// 3. POST /api/galleries — Create Event
// -------------------------------------------------------------
router.post('/', requireAuth, async (req, res) => {
  try {
    const userId = req.user.id || req.user.userId;
    const { name } = req.body;
    if (!name || !name.trim()) return res.status(400).json({ success: false, message: 'Event name required.' });

    const eventName = name.trim();
    const eventFolder = cleanFolderName(eventName);
    const eventId = String(Date.now());
    const slug = `${eventFolder.toLowerCase()}-${Date.now().toString().slice(-4)}`;

    const eventDir = path.join(__dirname, '..', 'uploads', 'galleries', eventFolder);
    if (!fs.existsSync(eventDir)) {
      fs.mkdirSync(eventDir, { recursive: true });
    }

    const db = await getDb();
    await db.run(
      `INSERT INTO events (id, user_id, name, slug, photo_count) VALUES (?, ?, ?, ?, 0)`,
      [eventId, userId, eventName, slug]
    );

    return res.status(201).json({ success: true, gallery: { id: eventId, name: eventName, slug } });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Could not create event.' });
  }
});

// -------------------------------------------------------------
// 4. PUT /api/galleries/:id/rename (Placed ABOVE generic GET /:id)
// -------------------------------------------------------------
router.put('/:id/rename', requireAuth, async (req, res) => {
  try {
    const eventId = String(req.params.id || '').trim();
    const userId = req.user.id || req.user.userId;
    const rawName = req.body.name || req.body.eventName;

    if (!rawName || !rawName.trim()) {
      return res.status(400).json({ success: false, message: 'A valid event name is required.' });
    }

    const newName = rawName.trim();
    const db = await getDb();

    // Verify ownership
    const event = await db.get('SELECT id, name FROM events WHERE id = ? AND user_id = ?', [eventId, userId]);
    if (!event) {
      return res.status(404).json({ success: false, message: 'Event not found or unauthorized.' });
    }

    // Update event name
    await db.run('UPDATE events SET name = ? WHERE id = ?', [newName, eventId]);

    // Invalidate in-memory cache
    invalidateEventCache(eventId);

    return res.status(200).json({ 
      success: true, 
      message: 'Event renamed successfully.', 
      name: newName 
    });
  } catch (err) {
    console.error('[Rename Event Error]:', err);
    return res.status(500).json({ success: false, message: 'Failed to rename event: ' + err.message });
  }
});

// -------------------------------------------------------------
// 5. GET /api/galleries/:id — Public Event Details & Photos
// -------------------------------------------------------------
router.get('/:id', async (req, res) => {
  try {
    const identifier = (req.params.id || '').trim();
    const db = await getDb();
    const event = await db.get(
      `SELECT * FROM events WHERE id = ? OR LOWER(slug) = LOWER(?)`,
      [identifier, identifier]
    );

    if (!event) return res.status(404).json({ success: false, message: 'Event not found.' });

    const photos = await db.all(
      `SELECT id, file_path as url, filename, folder_name, created_at 
       FROM photos 
       WHERE event_id = ? 
       ORDER BY created_at DESC`,
      [event.id]
    );

    return res.status(200).json({
      success: true,
      id: event.id,
      name: event.name,
      event_name: event.name,
      slug: event.slug,
      photo_count: photos.length,
      photos,
      event
    });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Error retrieving event.' });
  }
});

// -------------------------------------------------------------
// 6. POST /api/galleries/:id/upload-photos
// -------------------------------------------------------------
const handlePhotoUpload = async (req, res) => {
  try {
    const galleryId = req.params.id;
    const userId = req.user.id || req.user.userId;
    const db = await getDb();

    const event = await db.get('SELECT id, name FROM events WHERE id = ? AND user_id = ?', [galleryId, userId]);
    if (!event) return res.status(404).json({ success: false, message: 'Event not found.' });

    req.eventCleanName = cleanFolderName(event.name);

    uploadPhotosMiddleware(req, res, async (err) => {
      if (err) return res.status(400).json({ success: false, message: err.message });

      const files = req.files || [];
      const eventDir = req.eventCleanName;
      const subFolder = req.savedSubFolder || 'All_Photos';
      const newlyAddedPhotos = [];

      for (const file of files) {
        const photoId = `pho_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
        const relativeUrl = `/uploads/galleries/${eventDir}/${subFolder}/${file.filename}`;

        await db.run(
          `INSERT INTO photos (id, event_id, filename, file_path, user_id, folder_name) 
           VALUES (?, ?, ?, ?, ?, ?)`,
          [photoId, galleryId, file.filename, relativeUrl, userId, subFolder]
        );

        newlyAddedPhotos.push({
          id: photoId,
          fullPath: path.resolve(__dirname, '..', 'uploads', 'galleries', eventDir, subFolder, file.filename)
        });
      }

      await db.run(
        `UPDATE events SET photo_count = (SELECT COUNT(*) FROM photos WHERE event_id = ?) WHERE id = ?`,
        [galleryId, galleryId]
      );

      invalidateEventCache(galleryId);

      // Async background auto-index
      (async () => {
        try {
          const indexResult = await callAiDaemon('index-photos', { photos: newlyAddedPhotos });
          if (indexResult && indexResult.indexed) {
            for (const item of indexResult.indexed) {
              const embedId = `emb_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
              await db.run(
                `INSERT OR REPLACE INTO face_embeddings (id, event_id, photo_id, user_id, embedding) VALUES (?, ?, ?, ?, ?)`,
                [embedId, galleryId, item.photoId, userId, JSON.stringify(item.vector)]
              );
            }
            invalidateEventCache(galleryId);
          }
        } catch (idxErr) {
          console.error('[Async Indexing Error]:', idxErr.message);
        }
      })();

      return res.status(200).json({ success: true, message: `Saved ${files.length} photos.` });
    });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Upload failed.' });
  }
};

router.post('/:id/upload-photos', requireAuth, handlePhotoUpload);
router.post('/:id/upload', requireAuth, handlePhotoUpload);

// -------------------------------------------------------------
// 7. POST /api/galleries/:id/upload-zip
// -------------------------------------------------------------
router.post('/:id/upload-zip', requireAuth, (req, res) => {
  zipUpload(req, res, async (err) => {
    if (err || !req.file) return res.status(400).json({ success: false, message: 'Invalid ZIP file.' });

    const zipPath = req.file.path;
    const galleryId = req.params.id;
    const userId = req.user.id || req.user.userId;

    try {
      const db = await getDb();
      const event = await db.get('SELECT id, name FROM events WHERE id = ? AND user_id = ?', [galleryId, userId]);
      if (!event) {
        if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);
        return res.status(404).json({ success: false, message: 'Event not found.' });
      }

      const eventDir = cleanFolderName(event.name);
      const targetBase = path.join(__dirname, '..', 'uploads', 'galleries', eventDir);
      const validExtensions = ['.jpg', '.jpeg', '.png', '.webp'];
      const newlyAddedPhotos = [];

      await fs.createReadStream(zipPath)
        .pipe(unzipper.Parse())
        .on('entry', async (entry) => {
          const entryPath = entry.path;
          const ext = path.extname(entryPath).toLowerCase();

          if (entry.type === 'File' && validExtensions.includes(ext) && !entryPath.includes('__MACOSX')) {
            const parts = entryPath.split('/');
            const subFolder = parts.length > 1 ? cleanFolderName(parts[0]) : 'All_Photos';
            const fileName = path.basename(entryPath);
            const cleanName = `zip_${Date.now()}_${fileName}`;

            const fullDirPath = path.join(targetBase, subFolder);
            if (!fs.existsSync(fullDirPath)) {
              fs.mkdirSync(fullDirPath, { recursive: true });
            }

            const writeDest = path.join(fullDirPath, cleanName);
            entry.pipe(fs.createWriteStream(writeDest)).on('finish', async () => {
              const photoId = `pho_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
              const relativeUrl = `/uploads/galleries/${eventDir}/${subFolder}/${cleanName}`;

              await db.run(
                `INSERT INTO photos (id, event_id, filename, file_path, user_id, folder_name) 
                 VALUES (?, ?, ?, ?, ?, ?)`,
                [photoId, galleryId, cleanName, relativeUrl, userId, subFolder]
              );

              newlyAddedPhotos.push({ id: photoId, fullPath: writeDest });
            });
          } else {
            entry.autodrain();
          }
        })
        .promise();

      if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);

      await db.run(
        `UPDATE events SET photo_count = (SELECT COUNT(*) FROM photos WHERE event_id = ?) WHERE id = ?`,
        [galleryId, galleryId]
      );

      invalidateEventCache(galleryId);

      setTimeout(async () => {
        try {
          const indexResult = await callAiDaemon('index-photos', { photos: newlyAddedPhotos });
          if (indexResult && indexResult.indexed) {
            for (const item of indexResult.indexed) {
              const embedId = `emb_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
              await db.run(
                `INSERT OR REPLACE INTO face_embeddings (id, event_id, photo_id, user_id, embedding) VALUES (?, ?, ?, ?, ?)`,
                [embedId, galleryId, item.photoId, userId, JSON.stringify(item.vector)]
              );
            }
            invalidateEventCache(galleryId);
          }
        } catch (idxErr) {
          console.error('[Async Indexing Error]:', idxErr.message);
        }
      }, 1000);

      return res.status(200).json({ success: true, message: `Extracted photos into ${eventDir}` });
    } catch (zipErr) {
      if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);
      return res.status(500).json({ success: false, message: 'ZIP extraction failed.' });
    }
  });
});

// -------------------------------------------------------------
// 8. SUB-3-SECOND GUEST FACE SEARCH (RAM Cache Match)
// -------------------------------------------------------------
router.post('/:id/search-face', (req, res) => {
  selfieUpload(req, res, async (err) => {
    if (err) return res.status(400).json({ success: false, message: 'Selfie upload error: ' + err.message });

    const selfieFile = req.file;
    if (!selfieFile) return res.status(400).json({ success: false, message: 'No selfie image provided.' });

    const identifier = (req.params.id || '').trim();
    const t0 = Date.now();

    try {
      const db = await getDb();
      
      const event = await db.get(
        'SELECT id, name, slug FROM events WHERE id = ? OR LOWER(slug) = LOWER(?)',
        [identifier, identifier]
      );

      if (!event) {
        if (fs.existsSync(selfieFile.path)) fs.unlinkSync(selfieFile.path);
        return res.status(404).json({ success: false, message: `Event '${identifier}' not found.` });
      }

      console.log(`\n[AI SCAN] Starting fast face scan for: ${event.name}`);

      const selfieResult = await callAiDaemon('extract-selfie', {
        selfiePath: selfieFile.path
      });

      if (fs.existsSync(selfieFile.path)) fs.unlinkSync(selfieFile.path);

      if (!selfieResult || selfieResult.status !== 'success' || !selfieResult.embedding) {
        return res.status(400).json({ 
          success: false, 
          message: 'No clear face detected in reference photo. Please try a clearer portrait.' 
        });
      }

      const selfieVec = new Float32Array(selfieResult.embedding);
      const records = await getEventVectors(db, event.id, event.slug);

      if (records.length === 0) {
        return res.status(200).json({ 
          success: true, 
          event: { id: event.id, slug: event.slug, name: event.name },
          matchedPhotos: [] 
        });
      }

      const matchedMap = new Map();
      const PRIMARY_THRESHOLD = 0.42;

      for (let i = 0; i < records.length; i++) {
        const item = records[i];
        const v = item.vector;
        let score = 0.0;
        
        for (let j = 0; j < 512; j++) {
          score += selfieVec[j] * v[j];
        }

        if (score >= PRIMARY_THRESHOLD) {
          if (!matchedMap.has(item.photo_id) || matchedMap.get(item.photo_id).score < score) {
            matchedMap.set(item.photo_id, {
              id: item.photo_id,
              url: item.url,
              filename: item.filename,
              folder_name: item.folder_name,
              score: Math.round(score * 100) / 100
            });
          }
        }
      }

      const matchedPhotos = Array.from(matchedMap.values()).sort((a, b) => b.score - a.score);
      const totalSeconds = ((Date.now() - t0) / 1000).toFixed(2);
      
      console.log(`[AI SCAN] Match finished in ${totalSeconds}s! Found ${matchedPhotos.length} matching photos.`);

      return res.status(200).json({
        success: true,
        event: { id: event.id, slug: event.slug, name: event.name },
        matchedPhotos
      });

    } catch (err) {
      if (selfieFile && fs.existsSync(selfieFile.path)) fs.unlinkSync(selfieFile.path);
      console.error('[Face Search Error]:', err.message);
      return res.status(500).json({ success: false, message: 'AI scanning error: ' + err.message });
    }
  });
});

// -------------------------------------------------------------
// 9. POST /api/galleries/:id/download-zip (AdmZip Engine)
// -------------------------------------------------------------
router.post('/:id/download-zip', async (req, res) => {
  try {
    const identifier = (req.params.id || '').trim();
    const photoIds = req.body && Array.isArray(req.body.photoIds) ? req.body.photoIds : null;

    const db = await getDb();
    const event = await db.get(
      'SELECT id, name, slug FROM events WHERE id = ? OR LOWER(slug) = LOWER(?)',
      [identifier, identifier]
    );

    if (!event) {
      return res.status(404).json({ success: false, message: 'Event not found.' });
    }

    let photos = [];
    if (photoIds && photoIds.length > 0) {
      const placeholders = photoIds.map(() => '?').join(',');
      photos = await db.all(
        `SELECT id, file_path, filename FROM photos WHERE id IN (${placeholders})`,
        photoIds
      );
    } else {
      photos = await db.all(
        `SELECT id, file_path, filename FROM photos WHERE event_id = ?`,
        [event.id]
      );
    }

    if (!photos || photos.length === 0) {
      return res.status(400).json({ success: false, message: 'No photos available for download.' });
    }

    const projectRoot = path.resolve(__dirname, '..');
    const validFiles = [];

    for (const p of photos) {
      if (!p.file_path) continue;

      const rawRel = p.file_path.replace(/^[/\\]+/, '');
      const candidatePaths = [
        path.resolve(projectRoot, rawRel),
        path.resolve(projectRoot, 'uploads', rawRel.replace(/^uploads[/\\]/, '')),
        path.resolve(projectRoot, p.file_path)
      ];

      const foundPath = candidatePaths.find(cand => fs.existsSync(cand));

      if (foundPath) {
        validFiles.push({
          diskPath: foundPath,
          archiveName: p.filename || path.basename(foundPath)
        });
      } else {
        console.warn(`[ZIP SKIP] Photo not found on disk: ${p.file_path}`);
      }
    }

    if (validFiles.length === 0) {
      return res.status(404).json({ 
        success: false, 
        message: 'The requested photo files could not be found on server storage.' 
      });
    }

    const zip = new AdmZip();

    for (const fileItem of validFiles) {
      zip.addLocalFile(fileItem.diskPath, '', fileItem.archiveName);
    }

    const zipBuffer = zip.toBuffer();
    const safeName = cleanFolderName(event.name);
    const zipFileName = `${safeName}_${photoIds ? 'Matched' : 'All'}_${Date.now()}.zip`;

    res.setHeader('Content-Type', 'application/zip');
    res.setHeader('Content-Disposition', `attachment; filename="${zipFileName}"`);
    res.setHeader('Content-Length', zipBuffer.length);

    return res.end(zipBuffer);

  } catch (err) {
    console.error('[Download ZIP Error]:', err);
    if (!res.headersSent) {
      return res.status(500).json({ success: false, message: 'Failed to create download ZIP: ' + err.message });
    }
  }
});

// -------------------------------------------------------------
// 10. DELETE /api/galleries/:id
// -------------------------------------------------------------
router.delete('/:id', requireAuth, async (req, res) => {
  try {
    const galleryId = req.params.id;
    const userId = req.user.id || req.user.userId;
    const db = await getDb();

    const event = await db.get('SELECT id, name FROM events WHERE id = ? AND user_id = ?', [galleryId, userId]);
    if (!event) return res.status(404).json({ success: false, message: 'Event not found.' });

    const eventDir = cleanFolderName(event.name);
    const targetDir = path.join(__dirname, '..', 'uploads', 'galleries', eventDir);

    invalidateEventCache(galleryId);

    await db.run(`DELETE FROM face_embeddings WHERE event_id = ?`, [galleryId]);
    await db.run(`DELETE FROM photos WHERE event_id = ?`, [galleryId]);
    await db.run(`DELETE FROM events WHERE id = ?`, [galleryId]);

    if (fs.existsSync(targetDir)) fs.rmSync(targetDir, { recursive: true, force: true });

    return res.status(200).json({ success: true, message: `Deleted event: ${eventDir}` });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Delete failed.' });
  }
});

module.exports = router;