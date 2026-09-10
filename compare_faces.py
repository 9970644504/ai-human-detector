import sys
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis


def load_model():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    return app


def get_embedding(app, image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    faces = app.get(image)

    if len(faces) == 0:
        raise ValueError(f"No face detected: {image_path}")

    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0])
        * (f.bbox[3] - f.bbox[1])
    )

    return face.embedding


def cosine_similarity(a, b):
    return float(
        np.dot(a, b) /
        (np.linalg.norm(a) * np.linalg.norm(b))
    )


def main():
    try:
        if len(sys.argv) < 3:
            raise ValueError(
                "Usage: python compare_faces.py <image1> <image2>"
            )

        image1 = sys.argv[1]
        image2 = sys.argv[2]

        app = load_model()

        embedding1 = get_embedding(app, image1)
        embedding2 = get_embedding(app, image2)

        similarity = cosine_similarity(
            embedding1,
            embedding2
        )

        print(
            json.dumps(
                {
                    "success": True,
                    "image1": image1,
                    "image2": image2,
                    "similarity": similarity
                },
                indent=2
            )
        )

    except Exception as error:
        print(
            json.dumps(
                {
                    "success": False,
                    "error": str(error)
                },
                indent=2
            )
        )

        sys.exit(1)


if __name__ == "__main__":
    main()