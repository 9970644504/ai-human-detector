const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const multer = require('multer');
const fs = require('fs');
const path = require('path');
const getDb = require('../config/db');
const { JWT_SECRET, requireAuth } = require('../middleware/authMiddleware');

const router = express.Router();

// -------------------------------------------------------------
// Avatar upload storage — one file per user, old one replaced
// -------------------------------------------------------------
const avatarsDir = path.join(__dirname, '..', 'uploads', 'avatars');
if (!fs.existsSync(avatarsDir)) {
  fs.mkdirSync(avatarsDir, { recursive: true });
}

const avatarUpload = multer({
  storage: multer.diskStorage({
    destination: (req, file, cb) => cb(null, avatarsDir),
    filename: (req, file, cb) => {
      const userId = req.user.id || req.user.userId;
      const ext = path.extname(file.originalname).toLowerCase() || '.jpg';
      cb(null, `${userId}_${Date.now()}${ext}`);
    }
  }),
  limits: { fileSize: 8 * 1024 * 1024 },
  fileFilter: (req, file, cb) => {
    const ok = ['image/jpeg', 'image/png', 'image/webp'].includes(file.mimetype);
    cb(ok ? null : new Error('Only JPEG, PNG or WEBP images are allowed.'), ok);
  }
}).single('avatar');

// Helper cookie settings for cross-page compatibility
const COOKIE_OPTIONS = {
  httpOnly: false, // accessible to frontend scripts if needed
  sameSite: 'lax',
  maxAge: 7 * 24 * 60 * 60 * 1000 // 7 days
};

// -------------------------------------------------------------
// POST /api/auth/register — Register Photographer
// -------------------------------------------------------------
router.post('/register', async (req, res) => {
  try {
    const { name, email, password, role } = req.body;

    if (!name || !email || !password) {
      return res.status(400).json({ success: false, message: 'All fields are required.' });
    }

    if (password.length < 6) {
      return res.status(400).json({ success: false, message: 'Password must be at least 6 characters long.' });
    }

    const db = await getDb();
    const cleanEmail = email.trim().toLowerCase();
    const cleanName = name.trim();
    const userRole = role || 'photographer';

    // Verify unique email
    const existingUser = await db.get('SELECT id FROM users WHERE email = ?', [cleanEmail]);
    if (existingUser) {
      return res.status(400).json({ success: false, message: 'Email already registered.' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    const userId = Date.now().toString();

    // Insert user into SQLite (handles tables with or without the 'role' column)
    try {
      await db.run(
        'INSERT INTO users (id, name, email, password, role) VALUES (?, ?, ?, ?, ?)',
        [userId, cleanName, cleanEmail, hashedPassword, userRole]
      );
    } catch (dbErr) {
      if (dbErr.message && dbErr.message.includes('no column named role')) {
        await db.run(
          'INSERT INTO users (id, name, email, password) VALUES (?, ?, ?, ?)',
          [userId, cleanName, cleanEmail, hashedPassword]
        );
      } else {
        throw dbErr;
      }
    }

    const tokenPayload = {
      id: userId,
      userId: userId,
      email: cleanEmail,
      name: cleanName,
      role: userRole
    };

    // Sign using the shared JWT_SECRET from authMiddleware
    const token = jwt.sign(tokenPayload, JWT_SECRET, { expiresIn: '7d' });

    // Set cookie for middleware fallback
    res.cookie('token', token, COOKIE_OPTIONS);
    res.cookie('auth_token', token, COOKIE_OPTIONS);

    return res.status(201).json({
      success: true,
      message: 'Photographer account registered successfully.',
      token,
      user: { id: userId, name: cleanName, email: cleanEmail, role: userRole, avatar_url: null }
    });

  } catch (err) {
    console.error('[Auth Register Error]:', err);
    return res.status(500).json({ success: false, message: 'Registration failed: ' + err.message });
  }
});

// -------------------------------------------------------------
// POST /api/auth/login — Login Photographer
// -------------------------------------------------------------
router.post('/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    if (!email || !password) {
      return res.status(400).json({ success: false, message: 'Email and password are required.' });
    }

    const db = await getDb();
    const cleanEmail = email.trim().toLowerCase();

    const user = await db.get('SELECT * FROM users WHERE email = ?', [cleanEmail]);
    if (!user) {
      return res.status(400).json({ success: false, message: 'Invalid email or password.' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ success: false, message: 'Invalid email or password.' });
    }

    const assignedRole = user.role || 'photographer';
    const tokenPayload = {
      id: user.id,
      userId: user.id,
      email: user.email,
      name: user.name,
      role: assignedRole
    };

    // Sign using the shared JWT_SECRET from authMiddleware
    const token = jwt.sign(tokenPayload, JWT_SECRET, { expiresIn: '7d' });

    // Set cookie for middleware fallback
    res.cookie('token', token, COOKIE_OPTIONS);
    res.cookie('auth_token', token, COOKIE_OPTIONS);

    return res.status(200).json({
      success: true,
      message: 'Login successful.',
      token,
      user: { id: user.id, name: user.name, email: user.email, role: assignedRole, avatar_url: user.avatar_url || null }
    });

  } catch (err) {
    console.error('[Auth Login Error]:', err);
    return res.status(500).json({ success: false, message: 'Login failed: ' + err.message });
  }
});

// -------------------------------------------------------------
// GET /api/auth/me — Validate Active Session Token
// -------------------------------------------------------------
router.get('/me', requireAuth, async (req, res) => {
  try {
    const db = await getDb();
    const userId = req.user.id || req.user.userId;
    const user = await db.get('SELECT id, name, email, role, avatar_url, created_at FROM users WHERE id = ?', [userId]);

    if (!user) {
      return res.status(404).json({ success: false, message: 'User session not found.' });
    }

    return res.status(200).json({ success: true, user });
  } catch (err) {
    return res.status(500).json({ success: false, message: 'Failed to verify session.' });
  }
});

// -------------------------------------------------------------
// POST /api/auth/avatar — Upload / replace profile photo
// -------------------------------------------------------------
router.post('/avatar', requireAuth, (req, res) => {
  avatarUpload(req, res, async (uploadErr) => {
    if (uploadErr) {
      return res.status(400).json({ success: false, message: uploadErr.message || 'Avatar upload failed.' });
    }
    if (!req.file) {
      return res.status(400).json({ success: false, message: 'No image file received.' });
    }

    try {
      const db = await getDb();
      const userId = req.user.id || req.user.userId;

      // Remove the previous avatar file from disk, if any, so they don't pile up
      const existing = await db.get('SELECT avatar_url FROM users WHERE id = ?', [userId]);
      if (existing && existing.avatar_url) {
        const oldPath = path.join(__dirname, '..', existing.avatar_url.replace(/^\/+/, ''));
        if (fs.existsSync(oldPath)) {
          try { fs.unlinkSync(oldPath); } catch (e) {}
        }
      }

      const avatarUrl = `/uploads/avatars/${req.file.filename}`;
      await db.run('UPDATE users SET avatar_url = ? WHERE id = ?', [avatarUrl, userId]);

      return res.status(200).json({ success: true, message: 'Profile photo updated.', avatar_url: avatarUrl });
    } catch (err) {
      console.error('[Avatar Upload Error]:', err);
      return res.status(500).json({ success: false, message: 'Failed to save profile photo.' });
    }
  });
});

// -------------------------------------------------------------
// POST /api/auth/logout — Sign Out
// -------------------------------------------------------------
router.post('/logout', (req, res) => {
  res.clearCookie('token');
  res.clearCookie('auth_token');
  return res.status(200).json({ success: true, message: 'Signed out successfully.' });
});

module.exports = router;