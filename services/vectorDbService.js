const { QdrantClient } = require('@qdrant/js-client-rest');

let isQdrantOnline = false;

// Disable compatibility check for local offline/development setups
const client = new QdrantClient({ 
  url: process.env.QDRANT_URL || 'http://localhost:6333',
  checkCompatibility: false 
});

const COLLECTION_NAME = 'face_embeddings';

/**
 * Ensures the Qdrant face embeddings collection exists if server is online.
 */
async function initVectorDb() {
  try {
    const collections = await client.getCollections();
    const exists = collections.collections.some(c => c.name === COLLECTION_NAME);

    if (!exists) {
      await client.createCollection(COLLECTION_NAME, {
        vectors: { size: 512, distance: 'Cosine' }
      });
      console.log(`✅ Qdrant collection '${COLLECTION_NAME}' created.`);
    }
    isQdrantOnline = true;
    console.log('✅ Qdrant Vector Database connected.');
  } catch (err) {
    isQdrantOnline = false;
    console.log('ℹ️ Running in SQLite fallback mode (Qdrant offline).');
  }
}

/**
 * Inserts or updates face embedding vectors into Qdrant.
 */
async function upsertFaceVectors(points) {
  if (!isQdrantOnline) return;

  try {
    await client.upsert(COLLECTION_NAME, {
      wait: true,
      points: points.map(p => ({
        id: p.id,
        vector: p.vector,
        payload: {
          gallery_id: p.galleryId,
          photo_url: p.photoUrl
        }
      }))
    });
  } catch (err) {
    console.error('Error inserting vectors to Qdrant:', err.message);
  }
}

/**
 * Searches Qdrant for matching face vectors within a specific gallery.
 */
async function searchMatchingFaces(searchVector, galleryId, limit = 50, scoreThreshold = 0.45) {
  if (!isQdrantOnline) return [];

  try {
    const results = await client.query(COLLECTION_NAME, {
      query: searchVector,
      limit: limit,
      score_threshold: scoreThreshold,
      filter: {
        must: [
          {
            key: 'gallery_id',
            match: { value: galleryId }
          }
        ]
      }
    });

    const points = results.points || [];
    return points.map(hit => ({
      photoUrl: hit.payload.photo_url,
      score: hit.score
    }));
  } catch (err) {
    return [];
  }
}

module.exports = {
  initVectorDb,
  upsertFaceVectors,
  searchMatchingFaces
};