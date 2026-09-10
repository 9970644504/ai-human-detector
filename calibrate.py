import sys
import os
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


def get_embeddings(app, image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read: {image_path}"
        )

    faces = app.get(image)

    return [
        face.embedding
        for face in faces
    ]


def similarity(a, b):

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


def main():

    if len(sys.argv) < 3:

        print(
            "Usage: python calibrate.py "
            "<reference> <image1> [image2] ..."
        )

        sys.exit(1)

    reference_path = sys.argv[1]

    image_paths = sys.argv[2:]

    app = load_model()

    reference_faces = get_embeddings(
        app,
        reference_path
    )

    if not reference_faces:

        raise ValueError(
            "No face detected in reference."
        )

    reference = reference_faces[0]

    print()
    print("=" * 60)
    print("FACE SIMILARITY CALIBRATION")
    print("=" * 60)

    for image_path in image_paths:

        faces = get_embeddings(
            app,
            image_path
        )

        if not faces:

            print(
                f"\nNO FACE: {image_path}"
            )

            continue

        scores = [
            similarity(
                reference,
                face
            )
            for face in faces
        ]

        best = max(scores)

        print()
        print(
            f"Image: {image_path}"
        )

        print(
            f"Faces: {len(faces)}"
        )

        print(
            f"Similarities: "
            f"{[round(x, 4) for x in scores]}"
        )

        print(
            f"Best similarity: {best:.4f}"
        )

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
