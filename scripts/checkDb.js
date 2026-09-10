const path = require('path');
const fs = require('fs');
const getDb = require('../config/db');

(async () => {
  try {
    console.log('--- Connecting to SQLite Database ---');
    const db = await getDb();

    // 1. Inspect table columns
    const columns = await db.all("PRAGMA table_info(photos)");
    console.log('Photos table columns:', columns.map(c => c.name).join(', '));

    // 2. Fetch photos
    const photos = await db.all('SELECT * FROM photos');
    console.log(`Total photos found in table: ${photos.length}`);

    if (photos.length === 0) {
      console.log('\n[WARNING] No photos exist in the database! Upload photos through the workspace dashboard first.');
      process.exit(0);
    }

    console.log('\n--- EMBEDDINGS AUDIT ---');
    photos.forEach((p, idx) => {
      let faces = [];
      try {
        faces = JSON.parse(p.face_embeddings || '[]');
      } catch (e) {
        faces = [];
      }
      console.log(`${idx + 1}. Photo ID: ${p.id} | File: ${p.filename} | URL: ${p.url} | Faces Stored: ${faces.length}`);
    });
    console.log('------------------------\n');
  } catch (err) {
    console.error('DATABASE AUDIT ERROR:', err);
  } finally {
    process.exit(0);
  }
})();