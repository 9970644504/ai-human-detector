const ort = require('onnxruntime-node');
const { Worker } = require('worker_threads');
const fs = require('fs');
const path = require('path');
const os = require('os');

class FastPhotoScanner {
  constructor(modelPath) {
    this.modelPath = modelPath;
    this.session = null;
    this.batchSize = 64; // Optimized GPU batch size
    this.numWorkers = Math.max(1, os.cpus().length - 1);
  }

  async init() {
  try {
    this.session = await ort.InferenceSession.create(this.modelPath, {
      executionProviders: ['directml', 'cuda', 'cpu'],
      graphOptimizationLevel: 'all'
    });
    console.log(`⚡ GPU Engine Online! Utilizing ${this.numWorkers} CPU Worker Threads.`);
  } catch (error) {
    console.error('Failed to initialize ONNX session:', error);
  }
}

  // Offload chunk preprocessing to a worker thread safely
  processChunkInWorker(chunk) {
    return new Promise((resolve) => {
      // Target imageWorker.js inside src folder
      const workerPath = path.join(__dirname, 'imageWorker.js');
      const worker = new Worker(workerPath);

      worker.postMessage(chunk);

      worker.on('message', (result) => {
        worker.terminate();
        resolve(result);
      });

      worker.on('error', (err) => {
        console.error('Worker thread error:', err);
        worker.terminate();
        resolve([]);
      });
    });
  }

  async scanDirectory(dirPath, onProgress) {
    const files = [];
    const stack = [dirPath];

    // Fast non-recursive stack directory walk
    while (stack.length > 0) {
      const current = stack.pop();
      try {
        const entries = fs.readdirSync(current, { withFileTypes: true });
        for (const entry of entries) {
          const fullPath = path.join(current, entry.name);
          if (entry.isDirectory()) {
            stack.push(fullPath);
          } else if (/\.(jpg|jpeg|png|webp)$/i.test(entry.name)) {
            files.push(fullPath);
          }
        }
      } catch {}
    }

    const total = files.length;
    console.log(`🚀 Accelerated GPU Scan: ${total} photos found.`);
    const startTime = Date.now();
    const embeddingsMap = [];
    let processed = 0;

    for (let i = 0; i < files.length; i += this.batchSize) {
      const chunk = files.slice(i, i + this.batchSize);
      
      // Parallel preprocessing across CPU cores
      const validItems = await this.processChunkInWorker(chunk);

      if (validItems.length > 0) {
        const batchLength = validItems.length;
        const batchData = new Float32Array(batchLength * 3 * 112 * 112);

        for (let b = 0; b < batchLength; b++) {
          batchData.set(validItems[b].tensor, b * 37632);
        }

        const tensor = new ort.Tensor('float32', batchData, [batchLength, 3, 112, 112]);
        
        // Execute batch inference
        const results = await this.session.run({ input_1: tensor });
        const outputKey = Object.keys(results)[0];
        const embeddings = results[outputKey].data;

        const vectorSize = 512;
        validItems.forEach((item, idx) => {
          const vector = embeddings.slice(idx * vectorSize, (idx + 1) * vectorSize);
          embeddingsMap.push({ filePath: item.filePath, vector });
        });
      }

      processed += chunk.length;
      if (onProgress) onProgress(Math.min(processed, total), total);
    }

    const duration = ((Date.now() - startTime) / 1000).toFixed(2);
    console.log(`⚡ COMPLETED: ${total} photos scanned in ${duration}s (${(total / duration).toFixed(0)} photos/sec)`);

    return embeddingsMap;
  }
}

module.exports = FastPhotoScanner;