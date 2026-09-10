import sys
import cv2
import numpy as np
from insightface.app import FaceAnalysis


REFERENCE_IMAGE = r"C:\Users\dell\Pictures\person.jpg.jpg"


def cosine_similarity(a, b):
    denominator = np.linalg.norm(a) * np.linalg.norm(b)

    if denominator == 0:
        return 0.0

    return float(np.dot(a, b) / denominator)


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


def get_faces(app, image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read: {image_path}")

    return app.get(image)


def main():

    if len(sys.argv) < 2:
        print("Usage: python diagnose_misses.py <image1> <image2> ...")
        sys.exit(1)

    app = load_model()

    reference_faces = get_faces(
        app,
        REFERENCE_IMAGE
    )

    if not reference_faces:
        raise ValueError("No face found in reference image.")

    reference = reference_faces[0].embedding

    print()
    print("=" * 70)
    print("MISSED-FACE DIAGNOSTIC")
    print("=" * 70)

    for image_path in sys.argv[1:]:

        print()
        print("IMAGE:")
        print(image_path)

        faces = get_faces(
            app,
            image_path
        )

        print(f"Faces detected: {len(faces)}")

        if not faces:
            print("NO FACE DETECTED")
            continue

        scores = []

        for index, face in enumerate(faces, start=1):

            similarity = cosine_similarity(
                reference,
                face.embedding
            )

            scores.append(similarity)

            bbox = [
                int(x)
                for x in face.bbox
            ]

            print(
                f"Face {index}: "
                f"similarity={similarity:.4f} "
                f"box={bbox}"
            )

        print(
            f"BEST SIMILARITY: {max(scores):.4f}"
        )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()