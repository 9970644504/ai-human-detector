import os
import sys
import contextlib
import multiprocessing
from flask import Flask, request, jsonify

# Suppress engine verbose logs
os.environ['ORT_LOGGING_LEVEL'] = '3'
cores = str(multiprocessing.cpu_count())
os.environ["OMP_NUM_THREADS"] = cores
os.environ["MKL_NUM_THREADS"] = cores

with contextlib.redirect_stdout(sys.stderr):
    import cv2
    import numpy as np
    import insightface
    from insightface.app import FaceAnalysis

app = Flask(__name__)

print(f"[AI ENGINE] Initializing InsightFace on {cores} CPU threads...")

try:
    analyzer = FaceAnalysis(
        name='buffalo_sc', 
        allowed_modules=['detection', 'recognition'],
        providers=['CPUExecutionProvider']
    )
    analyzer.prepare(ctx_id=0, det_size=(320, 320))
    print("[AI ENGINE] Running ultra-fast buffalo_sc engine.")
except Exception as e:
    print(f"[AI ENGINE] Falling back to buffalo_l: {e}")
    analyzer = FaceAnalysis(
        name='buffalo_l', 
        allowed_modules=['detection', 'recognition'],
        providers=['CPUExecutionProvider']
    )
    analyzer.prepare(ctx_id=0, det_size=(224, 224))

print("[AI ENGINE] Ready on http://127.0.0.1:5001")

def fast_read(image_path, target_dim=380):
    if not os.path.exists(image_path):
        return None
    img = cv2.imread(image_path)
    if img is None:
        return None

    h, w = img.shape[:2]
    if max(h, w) > target_dim:
        scale = target_dim / float(max(h, w))
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    return img

@app.route('/extract-selfie', methods=['POST'])
def extract_selfie():
    data = request.get_json(force=True)
    selfie_path = data.get('selfiePath')
    
    # 380px max constraint guarantees sub-250ms CPU runtime
    img = fast_read(selfie_path, target_dim=380)
    if img is None:
        return jsonify({"status": "error", "message": "Could not read reference image"}), 400

    try:
        analyzer.det_model.input_size = (224, 224)
    except Exception:
        pass

    faces = analyzer.get(img)

    if not faces:
        try:
            analyzer.det_model.input_size = (320, 320)
            faces = analyzer.get(img)
        except Exception:
            pass

    if not faces:
        return jsonify({"status": "error", "message": "No face found in reference selfie."}), 400

    # Select primary (largest) face
    primary_face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    emb = primary_face.embedding
    norm = float(np.linalg.norm(emb))
    if norm == 0:
        return jsonify({"status": "error", "message": "Zero vector output"}), 400

    return jsonify({"status": "success", "embedding": (emb / norm).tolist()})

@app.route('/index-photos', methods=['POST'])
def index_photos():
    data = request.get_json(force=True)
    photos = data.get('photos', [])
    indexed = []

    try:
        analyzer.det_model.input_size = (480, 480)
    except Exception:
        pass

    for p in photos:
        img = fast_read(p["fullPath"], target_dim=960)
        if img is None:
            continue
        try:
            faces = analyzer.get(img)
            for f in faces:
                if hasattr(f, 'det_score') and f.det_score < 0.45:
                    continue
                emb = f.embedding
                norm = float(np.linalg.norm(emb))
                if norm > 0:
                    indexed.append({
                        "photoId": p["id"],
                        "vector": (emb / norm).tolist()
                    })
        except Exception as e:
            sys.stderr.write(f"Index error on {p['fullPath']}: {str(e)}\n")

    return jsonify({"status": "success", "indexed": indexed})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, threaded=True)