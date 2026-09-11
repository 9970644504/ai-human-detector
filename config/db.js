const sqlite3 = require('sqlite3').verbose();
const { open } = require('sqlite');
const path = require('path');
const fs = require('fs');

let dbInstance = null;

async function getDb() {
  if (dbInstance) {
    return dbInstance;
  }

  // Ensure data directory exists
  const dbFolder = path.join(__dirname, '..', 'data');
  if (!fs.existsSync(dbFolder)) {
    fs.mkdirSync(dbFolder, { recursive: true });
  }

  const dbPath = path.join(dbFolder, 'database.sqlite');

  dbInstance = await open({
    filename: dbPath,
    driver: sqlite3.Database
  });

  // Enable foreign keys and WAL mode for better concurrency
  await dbInstance.run('PRAGMA foreign_keys = ON;');
  await dbInstance.run('PRAGMA journal_mode = WAL;');

  // 1. Users table (Photographers)
  await dbInstance.exec(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      name TEXT NOT NULL,
      email TEXT UNIQUE NOT NULL,
      password TEXT NOT NULL,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP
    );
  `);

  // Migration: add columns that pre-existing users tables may predate
  try {
    await dbInstance.exec(`ALTER TABLE users ADD COLUMN avatar_url TEXT`);
  } catch (e) {
    // Column already exists — safe to ignore
  }
  try {
    await dbInstance.exec(`ALTER TABLE users ADD COLUMN role TEXT DEFAULT 'photographer'`);
  } catch (e) {
    // Column already exists — safe to ignore
  }

  // 2. Events / Galleries table (Multi-tenant scoped by user_id)
  await dbInstance.exec(`
    CREATE TABLE IF NOT EXISTS events (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL,
      name TEXT NOT NULL,
      slug TEXT UNIQUE,
      photo_count INTEGER DEFAULT 0,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
    );
  `);

  // 3. Photos table
  await dbInstance.exec(`
    CREATE TABLE IF NOT EXISTS photos (
      id TEXT PRIMARY KEY,
      event_id TEXT NOT NULL,
      user_id TEXT NOT NULL,
      filename TEXT NOT NULL,
      file_path TEXT NOT NULL,
      folder_name TEXT DEFAULT 'General',
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (event_id) REFERENCES events(id) ON DELETE CASCADE
    );
  `);

  // 4. Face Embeddings table (For ArcFace feature matching)
  await dbInstance.exec(`
    CREATE TABLE IF NOT EXISTS face_embeddings (
      id TEXT PRIMARY KEY,
      photo_id TEXT NOT NULL,
      event_id TEXT NOT NULL,
      user_id TEXT NOT NULL,
      embedding TEXT NOT NULL,
      box_x INTEGER,
      box_y INTEGER,
      box_w INTEGER,
      box_h INTEGER,
      created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
      FOREIGN KEY (photo_id) REFERENCES photos(id) ON DELETE CASCADE
    );
  `);

  return dbInstance;
}

module.exports = getDb;