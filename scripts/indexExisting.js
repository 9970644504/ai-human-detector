const path = require('path');
const fs = require('fs');
const getDb = require('../config/db');
const { indexPhotoFaces } = require('../utils/faceIndexer');

async function main() {
  console.log('--- Starting Indexing for All Photos ---');
  const db = await getDb();
  const photos = await db.all('SELECT * FROM photos');
  console.log(`Total photos to evaluate: ${photos.length}`);

  let successCount = 0;
  let skipCount = 0;

  for (let i = 0; i < photos.length; i++) {
    const p = photos[i];
    
    // Resolve file path
    let localPath = path.join(__dirname, '..', p.url.replace(/\\/g, '/'));
    if (!fs.existsSync(localPath)) {
      localPath = path.join(__dirname, '../uploads/galleries', String(p.event_id), p.filename);
    }

    if (!fs.existsSync(localPath)) {
      console.warn(`[${i + 1}/${photos.length}] File missing on disk: ${p.filename}`);
      continue;
    }

    console.log(`[${i + 1}/${photos.length}] Processing: ${p.filename}...`);
    try {
      await indexPhotoFaces(p.id, localPath);
      successCount++;
    } catch (err) {
      console.error(`Failed indexing ${p.filename}:`, err.message);
    }
  }

  console.log(`\n========================================`);
  console.log(`Indexing completed! Processed: ${successCount} photos.`);
  console.log(`========================================\n`);
  process.exit(0);
}

main().catch(err => {
  console.error('Fatal crash in indexExisting.js:', err);
  process.exit(1);
});