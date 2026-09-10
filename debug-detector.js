const path = require('path');
const { getEmbeddingsFromFile, initModels } = require('./utils/faceIndexer');

function computeCosineSimilarity(vecA, vecB) {
  let dotProduct = 0.0;
  for (let i = 0; i < vecA.length; i++) dotProduct += vecA[i] * vecB[i];
  return dotProduct;
}

async function run() {
  await initModels();

  const photoA = 'C:\\Users\\dell\\OneDrive\\Desktop\\ai-human-detector\\uploads\\galleries\\1788242456819\\zip-1788242645797\\Rohit Sharma\\Image_1.jpg';
  const photoB = 'C:\\Users\\dell\\OneDrive\\Desktop\\ai-human-detector\\uploads\\galleries\\1788242456819\\zip-1788242645797\\Rohit Sharma\\Image_1 (2).jpg';

  console.log('--- Extracting Faces ---');
  const embsA = await getEmbeddingsFromFile(photoA);
  const embsB = await getEmbeddingsFromFile(photoB);

  console.log(`Photo A faces: ${embsA.length} | Photo B faces: ${embsB.length}\n`);

  if (embsA.length === 0 || embsB.length === 0) {
    console.log('No faces found in one or both images.');
    return;
  }

  let highestScore = -Infinity;
  let bestPair = null;

  for (let i = 0; i < embsA.length; i++) {
    for (let j = 0; j < embsB.length; j++) {
      const score = computeCosineSimilarity(embsA[i], embsB[j]);
      console.log(`Pair (A[${i}] vs B[${j}]): Cosine = ${score.toFixed(4)}`);
      if (score > highestScore) {
        highestScore = score;
        bestPair = { a: i, b: j };
      }
    }
  }

  console.log(`\n===========================================`);
  console.log(`Best Face Match (A[${bestPair.a}] vs B[${bestPair.b}]): ${highestScore.toFixed(4)}`);
  console.log(`===========================================`);
}

run().catch(console.error);