const express = require('express');
const path = require('path');
const cors = require('cors');
const fs = require('fs');
const multer = require('multer');
const unzipper = require('unzipper');
const { spawn } = require('child_process');
const getDb = require('./config/db');

const app = express();
const PORT = process.env.PORT || 3000;

// Ensure necessary upload directories exist
const uploadBaseDir = path.join(__dirname, 'uploads');
const tempPhotosDir = path.join(__dirname, 'uploads', 'temp_photos');
const tempZipDir = path.join(__dirname, 'uploads', 'temp_zips');
const tempSelfieDir = path.join(__dirname, 'uploads', 'temp_selfies');
const avatarsDir = path.join(__dirname, 'uploads', 'avatars');

[uploadBaseDir, tempPhotosDir, tempZipDir, tempSelfieDir, avatarsDir].forEach((dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// -------------------------------------------------------------
// 0. AUTO-SPAWN PERSISTENT PYTHON AI WORKER (PORT 5001)
// -------------------------------------------------------------
let aiProcess = null;

function startAiEngine() {
  const scriptPath = path.join(__dirname, 'services', 'aiWorker.py');
  const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';

  console.log('[AI ENGINE] Spawning persistent background AI worker...');
  aiProcess = spawn(pythonCmd, [scriptPath], {
    stdio: 'inherit',
    windowsHide: true
  });

  aiProcess.on('error', (err) => {
    console.error('[AI ENGINE ERROR] Failed to spawn Python worker:', err.message);
  });

  aiProcess.on('exit', (code, signal) => {
    console.log(`[AI ENGINE] Python worker exited (Code: ${code}, Signal: ${signal})`);
  });
}

startAiEngine();

// Clean up Python process on server termination
const cleanupAIProcess = () => {
  if (aiProcess) {
    console.log('\n[SERVER] Shutting down AI worker process...');
    try {
      if (process.platform === 'win32') {
        spawn('taskkill', ['/pid', aiProcess.pid, '/f', '/t']);
      } else {
        aiProcess.kill('SIGTERM');
      }
    } catch (e) {}
  }
};

process.on('SIGINT', () => {
  cleanupAIProcess();
  process.exit(0);
});
process.on('SIGTERM', () => {
  cleanupAIProcess();
  process.exit(0);
});
process.on('exit', cleanupAIProcess);

// Helper function to send indexing requests to the persistent AI worker
async function triggerBackgroundIndexing(photos, eventId, userId) {
  if (!photos || photos.length === 0) return;
  try {
    const res = await fetch('http://127.0.0.1:5001/index-photos', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ photos })
    });

    if (!res.ok) return;

    const data = await res.json();
    if (data.status === 'success' && Array.isArray(data.indexed)) {
      const db = await getDb();
      for (const item of data.indexed) {
        const embedId = `emb_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
        await db.run(
          `INSERT OR REPLACE INTO face_embeddings (id, event_id, photo_id, user_id, embedding) 
           VALUES (?, ?, ?, ?, ?)`,
          [embedId, eventId, item.photoId, userId || 'default', JSON.stringify(item.vector)]
        );
      }
      console.log(`[AI INDEX] Successfully pre-indexed ${data.indexed.length} face vectors for Event ${eventId}`);
    }
  } catch (err) {
    console.warn(`[AI INDEX BACKGROUND ERROR]: ${err.message}`);
  }
}

// Initialize Database Tables on Startup
getDb()
  .then(async (db) => {
    console.log('[DB] SQLite database connected and verified.');
    await db.run(`
      CREATE TABLE IF NOT EXISTS face_embeddings (
        id TEXT PRIMARY KEY,
        event_id TEXT,
        photo_id TEXT,
        user_id TEXT,
        embedding TEXT
      )
    `);
  })
  .catch((err) => console.error('[DB Error] Initialization failed:', err.message));

// -------------------------------------------------------------
// 1. GLOBAL MIDDLEWARE
// -------------------------------------------------------------
app.use(cors());
app.use(express.json({ limit: '500mb' }));
app.use(express.urlencoded({ extended: true, limit: '500mb' }));

// Request logging (skips static asset spam)
app.use((req, res, next) => {
  if (!req.originalUrl.startsWith('/uploads') && !req.originalUrl.includes('.')) {
    console.log(`[REQ] ${req.method} ${req.originalUrl}`);
  }
  next();
});

// -------------------------------------------------------------
// 2. EXPLICIT PAGE NAVIGATION ROUTES
// -------------------------------------------------------------
app.get('/', (req, res) => {
  const workspacePage = path.join(__dirname, 'public', 'workspace.html');
  if (fs.existsSync(workspacePage)) return res.sendFile(workspacePage);
  return res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/workspace', (req, res) => {
  const workspacePage = path.join(__dirname, 'public', 'workspace.html');
  if (fs.existsSync(workspacePage)) return res.sendFile(workspacePage);
  return res.redirect('/');
});

app.get(['/login', '/login.html', '/register', '/portal'], (req, res) => {
  return res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get(['/events', '/collections'], (req, res) => {
  const eventsPage = path.join(__dirname, 'public', 'events.html');
  if (fs.existsSync(eventsPage)) return res.sendFile(eventsPage);
  return res.redirect('/workspace');
});

app.get(['/event-detail', '/event-details'], (req, res) => {
  const detailPage = path.join(__dirname, 'public', 'event-detail.html');
  if (fs.existsSync(detailPage)) return res.sendFile(detailPage);
  return res.redirect('/workspace');
});

// Public Guest Portal Routes (slug or ID)
app.get(['/slug/:slug', '/gallery/:slug', '/gallery'], (req, res) => {
  const galleryPage = path.join(__dirname, 'public', 'gallery.html');
  if (fs.existsSync(galleryPage)) return res.sendFile(galleryPage);
  return res.redirect('/');
});

// -------------------------------------------------------------
// 3. STATIC FILE HANDLERS
// -------------------------------------------------------------
app.use('/uploads', (req, res, next) => {
  res.setHeader('Cross-Origin-Resource-Policy', 'cross-origin');
  next();
}, express.static(path.join(__dirname, 'uploads')));

app.use(express.static(path.join(__dirname, 'public'), { index: false }));
app.use(express.static(__dirname, { index: false }));

// -------------------------------------------------------------
// 4. MOUNT API MODULAR ROUTERS
// -------------------------------------------------------------
const authRoutes = require('./routes/authRoutes');
const galleryRoutes = require('./routes/galleryRoutes');
const aiRoutes = require('./routes/aiRoutes');

if (authRoutes) app.use('/api/auth', authRoutes);
app.use('/api/galleries', galleryRoutes);
app.use('/api/events', galleryRoutes);
if (aiRoutes) app.use('/api/ai', aiRoutes);

// -------------------------------------------------------------
// 5. DIRECT UPLOAD FALLBACKS WITH AUTO-INDEXING
// -------------------------------------------------------------

// Folder / Loose Photos Batch Upload Handler
const folderUploadMulter = multer({
  dest: tempPhotosDir,
  limits: { fileSize: 100 * 1024 * 1024 }
}).array('photos', 500);

app.post('/api/galleries/:id/upload-photos', folderUploadMulter, async (req, res) => {
  try {
    const galleryId = req.params.id;
    const files = req.files || [];

    if (files.length === 0) {
      return res.status(400).json({ success: false, message: 'No images were received.' });
    }

    const db = await getDb();
    const gallery = await db.get('SELECT * FROM events WHERE id = ? OR slug = ?', [galleryId, galleryId]);
    if (!gallery) {
      return res.status(404).json({ success: false, message: 'Gallery event not found.' });
    }

    const userId = gallery.user_id || 'default';
    const targetDir = path.join(__dirname, 'uploads', 'galleries', String(userId), String(gallery.id));
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }

    const newlyAddedPhotos = [];

    for (const file of files) {
      const ext = path.extname(file.originalname).toLowerCase() || '.jpg';
      const newFilename = `img_${Date.now()}_${Math.random().toString(36).substring(2, 7)}${ext}`;
      const destPath = path.join(targetDir, newFilename);

      fs.renameSync(file.path, destPath);

      const photoId = `pho_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
      const relativeUrl = `/uploads/galleries/${userId}/${gallery.id}/${newFilename}`;

      await db.run(
        `INSERT INTO photos (id, event_id, filename, file_path, user_id, folder_name) 
         VALUES (?, ?, ?, ?, ?, ?)`,
        [photoId, gallery.id, newFilename, relativeUrl, userId, 'Folder Uploads']
      );

      newlyAddedPhotos.push({
        id: photoId,
        fullPath: destPath
      });
    }

    await db.run(
      `UPDATE events 
       SET photo_count = (SELECT COUNT(*) FROM photos WHERE event_id = ?) 
       WHERE id = ?`,
      [gallery.id, gallery.id]
    );

    // Auto-index photos in the background so face search is instantaneous
    triggerBackgroundIndexing(newlyAddedPhotos, gallery.id, userId);

    console.log(`[Folder Upload] Uploaded and saved ${files.length} photos to Event ${gallery.id}`);
    return res.status(200).json({
      success: true,
      message: `Successfully uploaded and saved ${files.length} photos!`,
      count: files.length
    });
  } catch (err) {
    console.error('[Direct Folder Upload Error]:', err);
    return res.status(500).json({ success: false, message: err.message });
  }
});

// ZIP Batch Extraction Handler
const zipUploadMulter = multer({
  dest: tempZipDir,
  limits: { fileSize: 500 * 1024 * 1024 }
}).single('zipFile');

app.post('/api/galleries/:id/upload-zip', zipUploadMulter, async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ success: false, message: 'No ZIP file received.' });
  }

  const zipPath = req.file.path;
  const galleryId = req.params.id;

  try {
    const db = await getDb();
    const gallery = await db.get('SELECT * FROM events WHERE id = ? OR slug = ?', [galleryId, galleryId]);

    if (!gallery) {
      if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);
      return res.status(404).json({ success: false, message: 'Gallery event not found.' });
    }

    const userId = gallery.user_id || 'default';
    const targetDir = path.join(__dirname, 'uploads', 'galleries', String(userId), String(gallery.id));
    if (!fs.existsSync(targetDir)) {
      fs.mkdirSync(targetDir, { recursive: true });
    }

    const validExtensions = ['.jpg', '.jpeg', '.png', '.webp'];
    const newlyAddedPhotos = [];

    await fs
      .createReadStream(zipPath)
      .pipe(unzipper.Parse())
      .on('entry', async (entry) => {
        const fileName = entry.path;
        const ext = path.extname(fileName).toLowerCase();

        if (entry.type === 'File' && validExtensions.includes(ext) && !fileName.includes('__MACOSX')) {
          const newName = `zip_${Date.now()}_${path.basename(fileName)}`;
          const destFile = path.join(targetDir, newName);

          entry.pipe(fs.createWriteStream(destFile)).on('finish', async () => {
            const photoId = `pho_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
            const relativeUrl = `/uploads/galleries/${userId}/${gallery.id}/${newName}`;

            await db.run(
              `INSERT INTO photos (id, event_id, filename, file_path, user_id, folder_name) 
               VALUES (?, ?, ?, ?, ?, ?)`,
              [photoId, gallery.id, newName, relativeUrl, userId, 'ZIP Archive']
            );

            newlyAddedPhotos.push({
              id: photoId,
              fullPath: destFile
            });
          });
        } else {
          entry.autodrain();
        }
      })
      .promise();

    if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);

    await db.run(
      `UPDATE events 
       SET photo_count = (SELECT COUNT(*) FROM photos WHERE event_id = ?) 
       WHERE id = ?`,
      [gallery.id, gallery.id]
    );

    // Auto-index the extracted ZIP photos in background
    setTimeout(() => {
      triggerBackgroundIndexing(newlyAddedPhotos, gallery.id, userId);
    }, 1000);

    console.log(`[ZIP Upload] Extracted files to Event ${gallery.id}`);
    return res.status(200).json({
      success: true,
      message: `ZIP archive processed and extracted successfully!`
    });
  } catch (err) {
    if (fs.existsSync(zipPath)) fs.unlinkSync(zipPath);
    console.error('[Direct ZIP Upload Error]:', err);
    return res.status(500).json({ success: false, message: err.message });
  }
});

// -------------------------------------------------------------
// 6. PHOTO DELETION ENDPOINTS
// -------------------------------------------------------------
const handleDeletePhoto = async (req, res) => {
  const { photoId } = req.params;

  try {
    const db = await getDb();
    
    // 1. Locate photo details in database
    const photo = await db.get('SELECT * FROM photos WHERE id = ?', [photoId]);
    if (!photo) {
      return res.status(404).json({ success: false, message: 'Photo not found.' });
    }

    const eventId = photo.event_id;

    // 2. Remove associated face embeddings
    await db.run('DELETE FROM face_embeddings WHERE photo_id = ?', [photoId]);

    // 3. Remove record from database
    await db.run('DELETE FROM photos WHERE id = ?', [photoId]);

    // 4. Safely delete file from disk if it exists
    if (photo.file_path) {
      let cleanRelPath = photo.file_path.replace(/^\/+/, '');
      let fullDiskPath = path.join(__dirname, cleanRelPath);
      
      // Secondary path fallback
      if (!fs.existsSync(fullDiskPath) && photo.filename) {
        fullDiskPath = path.join(__dirname, 'uploads', 'galleries', String(photo.user_id || 'default'), String(eventId), photo.filename);
      }

      if (fs.existsSync(fullDiskPath)) {
        try {
          fs.unlinkSync(fullDiskPath);
          console.log(`[PHOTO DELETE] File removed: ${fullDiskPath}`);
        } catch (fileErr) {
          console.warn(`[PHOTO DELETE] Could not delete file on disk: ${fileErr.message}`);
        }
      }
    }

    // 5. Update parent event's photo counter
    if (eventId) {
      await db.run(
        `UPDATE events 
         SET photo_count = (SELECT COUNT(*) FROM photos WHERE event_id = ?) 
         WHERE id = ?`,
        [eventId, eventId]
      );
    }

    console.log(`[PHOTO DELETE] Successfully deleted photo ${photoId} from Event ${eventId}`);
    return res.status(200).json({ success: true, message: 'Photo deleted successfully.' });
  } catch (err) {
    console.error('[PHOTO DELETE ERROR]:', err);
    return res.status(500).json({ success: false, message: err.message });
  }
};

// Supports both direct & gallery-nested DELETE requests
app.delete('/api/photos/:photoId', handleDeletePhoto);
app.delete('/api/galleries/:galleryId/photos/:photoId', handleDeletePhoto);

// -------------------------------------------------------------
// 7. USER SIGN OUT ENDPOINT
// -------------------------------------------------------------
app.post('/api/auth/logout', (req, res) => {
  res.clearCookie('token');
  res.clearCookie('auth_token');
  res.clearCookie('connect.sid');
  return res.status(200).json({ success: true, message: 'Signed out successfully.' });
});

// -------------------------------------------------------------
// 8. CATCH-ALL FOR UNMATCHED API ENDPOINTS
// -------------------------------------------------------------
app.use('/api', (req, res) => {
  console.warn(`[404 NOT FOUND] ${req.method} ${req.originalUrl}`);
  return res.status(404).json({ message: `API endpoint not found: ${req.originalUrl}` });
});

// -------------------------------------------------------------
// 9. CATCH-ALL FOR FRONTEND PAGE NAVIGATION
// -------------------------------------------------------------
app.use((req, res) => {
  const cleanPath = req.path.replace(/^\/+/, '');
  const publicPath = path.join(__dirname, 'public', cleanPath);
  const rootPath = path.join(__dirname, cleanPath);

  if (cleanPath && fs.existsSync(publicPath) && fs.statSync(publicPath).isFile()) {
    return res.sendFile(publicPath);
  } else if (cleanPath && fs.existsSync(rootPath) && fs.statSync(rootPath).isFile()) {
    return res.sendFile(rootPath);
  }

  const htmlPath = path.join(__dirname, 'public', `${cleanPath}.html`);
  if (cleanPath && fs.existsSync(htmlPath) && fs.statSync(htmlPath).isFile()) {
    return res.sendFile(htmlPath);
  }

  const defaultPage = path.join(__dirname, 'public', 'workspace.html');
  if (fs.existsSync(defaultPage)) {
    return res.sendFile(defaultPage);
  }
  return res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// -------------------------------------------------------------
// 10. START SERVER (Bound to 0.0.0.0 for LAN access)
// -------------------------------------------------------------
const server = app.listen(PORT, '0.0.0.0', () => {
  console.log(`\n=================================================`);
  console.log(`[LOCAL]  http://localhost:${PORT}`);
  console.log(`[LAN]    http://192.168.1.15:${PORT}`);
  console.log(`=================================================\n`);
});

server.ref();