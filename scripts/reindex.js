const path = require('path');
const fs = require('fs');
const getDb = require('../config/db');
const { indexPhotoFaces, initModels } = require('../utils/faceIndexer');

async function run() {
  await initModels();
  const db = await getDb();
  const photos = await db.all('SELECT id, event_id, filename FROM photos');

  console.log(`\nFound ${photos.length} photos in database to index...\n`);

  let count = 0;
  for (let i = 0; i < photos.length; i++) {
    const p = photos[i];
    const fullPath = path.join(__dirname, '..', 'uploads', 'galleries', String(p.event_id), p.filename);
    
    if (fs.existsSync(fullPath)) {
      console.log(`[${i + 1}/${photos.length}] Processing: ${p.filename}`);
      await indexPhotoFaces(p.id, fullPath);
      count++;
    } else {
      console.warn(`[Skip] File not found on disk: ${fullPath}`);
    }
  }

  console.log(`\n Successfully indexed ${count} photos! You can now run searches.\n`);
  process.exit(0);
}

run().catch(err => {
  console.error('Reindexing failed:', err);
  process.exit(1);
});
