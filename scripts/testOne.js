const path = require('path');
const fs = require('fs');

async function test() {
  console.log('--- Step 1: Testing imports ---');
  const getDb = require('../config/db');
  console.log('db config loaded');

  const { indexPhotoFaces } = require('../utils/faceIndexer');
  console.log('faceIndexer loaded');

  console.log('--- Step 2: Testing Database Query ---');
  const db = await getDb();
  const photo = await db.get('SELECT * FROM photos LIMIT 1');
  console.log('Sample photo record:', photo);

  if (!photo) {
    console.log('No photos found in database.');
    process.exit(0);
  }

  console.log('--- Step 3: Checking File On Disk ---');
  const filePath = path.join(__dirname, '..', photo.url.replace(/\\/g, '/'));
  console.log('Looking for file at:', filePath);
  console.log('File exists?', fs.existsSync(filePath));

  if (!fs.existsSync(filePath)) {
    // Try gallery path fallback
    const altPath = path.join(__dirname, '../uploads/galleries', String(photo.event_id), photo.filename);
    console.log('Testing fallback path:', altPath);
    console.log('Fallback exists?', fs.existsSync(altPath));
  }

  console.log('--- Step 4: Testing indexPhotoFaces() ---');
  const targetPath = fs.existsSync(filePath) 
    ? filePath 
    : path.join(__dirname, '../uploads/galleries', String(photo.event_id), photo.filename);

  await indexPhotoFaces(photo.id, targetPath);
  console.log('--- Finished testOne ---');
  process.exit(0);
}

test().catch(err => {
  console.error('CRASH IN testOne.js:', err);
  process.exit(1);
});