const { spawn } = require('child_process');
const path = require('path');

function runAiWorker(payload) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(__dirname, 'aiWorker.py');
    
    // On Windows, use 'python' or 'py'
    const pyProcess = spawn('python', [scriptPath]);

    let stdoutData = '';
    let stderrData = '';

    pyProcess.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString();
    });

    pyProcess.stderr.on('data', (chunk) => {
      stderrData += chunk.toString();
    });

    pyProcess.on('error', (err) => {
      reject(new Error(`Failed to start Python process: ${err.message}`));
    });

    pyProcess.on('close', (code) => {
      if (stderrData) {
        console.warn('[AI Worker Stderr]:', stderrData);
      }

      if (!stdoutData.trim()) {
        return reject(new Error('AI Worker returned empty output.'));
      }

      try {
        const parsed = JSON.parse(stdoutData.trim());
        resolve(parsed);
      } catch (parseErr) {
        console.error('[AI Worker Raw Output]:', stdoutData);
        reject(new Error(`Could not parse AI output: ${parseErr.message}`));
      }
    });

    // Write input JSON to stdin and close the stream
    pyProcess.stdin.write(JSON.stringify(payload));
    pyProcess.stdin.end();
  });
}

/**
 * Searches gallery photos using the uploaded selfie
 */
async function matchSelfieWithGallery(selfieAbsolutePath, galleryRelativeUrls, threshold = 0.40) {
  const payload = {
    action: 'match',
    selfiePath: selfieAbsolutePath,
    galleryPhotos: galleryRelativeUrls,
    threshold
  };

  return await runAiWorker(payload);
}

module.exports = {
  runAiWorker,
  matchSelfieWithGallery
};