const path = require('path');
const fs = require('fs');
const getDb = require('./config/db');
const { indexPhotoFaces, initModels } = require('./utils/faceIndexer');

// Helper: Sanitize string into directory name matching galleryRoutes.js
function sanitizeDirName(str) {
  if (!str) return 'General';
  return str
    .toString()
    .trim()
    .replace(/[<>:"/\\|?*]+/g, '-')
    .replace(/\s+/g, '_');
}

// Helper: Find file recursively within a folder
function findFileRecursive(dir, targetFilename) {
  if (!fs.existsSync(dir)) return null;
  const entries = fs.readdirSync(dir, { withFileTypes: true });

  for (const entry of entries) {
    const fullPath = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      const found = findFileRecursive(fullPath, targetFilename);
      if (found) return found;
    } else if (entry.name === targetFilename) {
      return fullPath;
    }
  }
  return null;
}

async function reindexAll() {
  console.log('[AI Engine] Initializing models...');
  await initModels();
  
  const db = await getDb();

  // Fetch all photos joined with event details to resolve named paths
  const photos = await db.all(`
    SELECT 
      photos.id, 
      photos.event_id, 
      photos.filename, 
      photos.url, 
      photos.folder_name,
      events.name AS event_name
    FROM photos
    LEFT JOIN events ON photos.event_id = events.id
  `);

  console.log(`Found ${photos.length} total photo(s) to re-index...\n`);

  let count = 0;
  let skipped = 0;

  for (const p of photos) {
    let resolvedPath = null;

    // 1. Primary Strategy: Try direct disk path matching the stored URL
    if (p.url && p.url.startsWith('/uploads/')) {
      const directPath = path.join(__dirname, p.url);
      if (fs.existsSync(directPath)) {
        resolvedPath = directPath;
      }
    }

    // 2. Secondary Strategy: Look inside uploads/galleries/<Event_Name>/
    if (!resolvedPath && p.event_name) {
      const namedEventDir = path.join(__dirname, 'uploads', 'galleries', sanitizeDirName(p.event_name));
      resolvedPath = findFileRecursive(namedEventDir, p.filename);
    }

    // 3. Fallback Strategy: Look inside legacy numeric ID directory uploads/galleries/<event_id>/
    if (!resolvedPath && p.event_id) {
      const idEventDir = path.join(__dirname, 'uploads', 'galleries', String(p.event_id));
      resolvedPath = findFileRecursive(idEventDir, p.filename);
    }

    // Process face re-indexing
    if (resolvedPath && fs.existsSync(resolvedPath)) {
      try {
        const numFaces = await indexPhotoFaces(p.id, resolvedPath);
        count++;
        console.log(`[${count}/${photos.length}] Indexed (${numFaces} face(s)): ${p.filename}`);
      } catch (err) {
        console.error(`[Error on photo ${p.id}]:`, err.message);
      }
    } else {
      skipped++;
      console.warn(`[Skip: Not Found on Disk] Photo ID: ${p.id} | Filename: ${p.filename}`);
    }
  }

  console.log(`\n======================================================`);
  console.log(`Re-indexing complete!`);
  console.log(`Successfully re-indexed: ${count}`);
  console.log(`Skipped / Not found:     ${skipped}`);
  console.log(`======================================================`);

  process.exit(0);
}

reindexAll().catch((err) => {
  console.error('[Fatal Re-index Error]:', err);
  process.exit(1);
});