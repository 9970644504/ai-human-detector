const { parentPort } = require('worker_threads');
const sharp = require('sharp');

sharp.cache(false);

parentPort.on('message', async (filePaths) => {
  const results = await Promise.all(
    filePaths.map(async (filePath) => {
      try {
        const { data } = await sharp(filePath)
          .resize(112, 112, { fit: 'fill', fastShrinkOnLoad: true })
          .toFormat('raw')
          .toBuffer({ resolveWithObject: true });

        // Fast Float32 conversion
        const floatData = new Float32Array(112 * 112 * 3);
        for (let i = 0; i < data.length; i++) {
          floatData[i] = (data[i] - 127.5) / 128.0;
        }
        return { filePath, tensor: floatData };
      } catch {
        return null;
      }
    })
  );

  parentPort.postMessage(results.filter(Boolean));
});