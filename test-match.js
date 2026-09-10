const path = require('path');
const { getEmbeddingsFromFile, initModels } = require('./utils/faceIndexer');

function computeCosineSimilarity(vecA, vecB) {
  let dotProduct = 0.0;
  for (let i = 0; i < vecA.length; i++) dotProduct += vecA[i] * vecB[i];
  return dotProduct;
}

async function runTest() {
  await initModels();

  // REPLACE THESE with real image paths on your machine
  const selfiePath = path.join(__dirname, 'test_selfie.jpg');
  const eventPhotoPath = path.join(__dirname, 'test_event.jpg');

  console.log('Extracting selfie embeddings...');
  const selfieEmbs = await getEmbeddingsFromFile(selfiePath);
  console.log(`Found ${selfieEmbs.length} face(s) in selfie.`);

  console.log('Extracting event photo embeddings...');
  const eventEmbs = await getEmbeddingsFromFile(eventPhotoPath);
  console.log(`Found ${eventEmbs.length} face(s) in event photo.`);

  if (selfieEmbs.length === 0 || eventEmbs.length === 0) {
    console.log('ERROR: Face detector failed to find faces in one of the test images.');
    return;
  }

  for (let i = 0; i < selfieEmbs.length; i++) {
    for (let j = 0; j < eventEmbs.length; j++) {
      const score = computeCosineSimilarity(selfieEmbs[i], eventEmbs[j]);
      console.log(`Cosine Similarity between Selfie Face #${i + 1} and Event Face #${j + 1}: ${score.toFixed(4)}`);
    }
  }
}

runTest();