const path = require('path');
const fs = require('fs');
const getDb = require('./config/db');
const { spawn } = require('child_process');

function runAiWorker(payload) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, 'services', 'aiWorker.py');
    const pyProcess = spawn('python', [scriptPath]);
    let stdoutData = '';
    let stderrData = '';

    pyProcess.stdout.on('data', (c) => (stdoutData += c.toString()));
    pyProcess.stderr.on('data', (c) => (stderrData += c.toString()));

    pyProcess.on('close', () => {
      try {
        const startTag = '__AI_OUT__';
        const endTag = '__AI_END__';
        const s = stdoutData.indexOf(startTag);
        const e = stdoutData.indexOf(endTag);
        if (s !== -1 && e !== -1) {
          return resolve(JSON.parse(stdoutData.substring(s + startTag.length, e).trim()));
        }

        // Fallback parser if tags were missed
        const lines = stdoutData.trim().split(/\r?\n/).reverse();
        for (const line of lines) {
          const trimmed = line.trim();
          if (trimmed.startsWith('{') && trimmed.endsWith('}')) {
            try { return resolve(JSON.parse(trimmed)); } catch (err) {}
          }
        }

        reject(new Error(stderrData || stdoutData || 'Empty output from AI worker'));
      } catch (err) {
        reject(err);
      }
    });

    pyProcess.stdin.write(JSON.stringify(payload));
    pyProcess.stdin.end();
  });
}

async function startPreIndexing() {
  const db = await getDb();

  // Fetch photos along with their event's user_id
  const photos = await db.all(`
    SELECT p.id, p.event_id, p.file_path, COALESCE(p.user_id, e.user_id, '1') AS user_id 
    FROM photos p
    LEFT JOIN events e ON p.event_id = e.id
  `);

  const stored = await db.all('SELECT photo_id FROM face_embeddings');
  const indexedIds = new Set(stored.map((r) => r.photo_id));

  const unindexed = photos.filter((p) => !indexedIds.has(p.id));
  console.log(`\nTotal photos: ${photos.length}. ${unindexed.length} pending indexing.`);

  if (unindexed.length === 0) {
    console.log('All photos are already indexed.');
    process.exit(0);
  }

  const BATCH_SIZE = 15;
  for (let i = 0; i < unindexed.length; i += BATCH_SIZE) {
    const chunk = unindexed.slice(i, i + BATCH_SIZE);

    // Normalize paths cleanly across Windows/POSIX slashes
    const payloadPhotos = chunk
      .map((p) => {
        const relativeClean = p.file_path.replace(/^[/\\]+/, '').replace(/[/\\]/g, path.sep);
        const absoluteDiskPath = path.resolve(__dirname, relativeClean);
        return {
          id: p.id,
          fullPath: absoluteDiskPath,
        };
      })
      .filter((p) => {
        const exists = fs.existsSync(p.fullPath);
        if (!exists) {
          console.warn(`[Missing File]: ${p.fullPath}`);
        }
        return exists;
      });

    console.log(
      `\nProcessing batch ${Math.floor(i / BATCH_SIZE) + 1} of ${Math.ceil(unindexed.length / BATCH_SIZE)} (${payloadPhotos.length} files on disk)...`
    );

    if (payloadPhotos.length === 0) {
      console.warn('Skipping batch: No files were found on disk.');
      continue;
    }

    try {
      const result = await runAiWorker({
        action: 'index_photos',
        photos: payloadPhotos,
      });

      if (result.status === 'success' && Array.isArray(result.indexed)) {
        for (const item of result.indexed) {
          const matchingPhoto = chunk.find((c) => c.id === item.photoId);
          const embedId = `emb_${Date.now()}_${Math.random().toString(36).substring(2, 7)}`;
          const fallbackUserId = matchingPhoto.user_id || '1';

          // Explicitly supply user_id to satisfy SQLITE NOT NULL constraint
          await db.run(
            `INSERT OR REPLACE INTO face_embeddings (id, event_id, photo_id, user_id, embedding) 
             VALUES (?, ?, ?, ?, ?)`,
            [embedId, matchingPhoto.event_id, item.photoId, fallbackUserId, JSON.stringify(item.vector)]
          );
        }
        console.log(`Saved ${result.indexed.length} face vector(s) for this batch.`);
      } else {
        console.warn('AI worker returned non-success or empty indexed array:', result);
      }
    } catch (batchErr) {
      console.error(`Batch ${Math.floor(i / BATCH_SIZE) + 1} processing error:`, batchErr.message);
    }
  }

  console.log('\nIndexing complete! Guest search will now execute in ~1 second.');
  process.exit(0);
}

startPreIndexing();