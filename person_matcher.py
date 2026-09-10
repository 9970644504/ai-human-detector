import sys
import json
import cv2
import numpy as np
from insightface.app import FaceAnalysis


# TEMPORARY threshold.
# We will calibrate this properly later.
MATCH_THRESHOLD = 0.80


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
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    faces = app.get(image)

    if len(faces) == 0:
        raise ValueError(
            f"No face detected: {image_path}"
        )

    # Use the largest detected face
    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0])
        * (f.bbox[3] - f.bbox[1])
    )

    return face.embedding


def cosine_similarity(a, b):

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            *
            np.linalg.norm(b)
        )
    )


def compare(reference_image, test_image):

    app = load_model()

    reference_embedding = get_embedding(
        app,
        reference_image
    )

    test_embedding = get_embedding(
        app,
        test_image
    )

    similarity = cosine_similarity(
        reference_embedding,
        test_embedding
    )

    matched = similarity >= MATCH_THRESHOLD

    return {
        "reference_image": reference_image,
        "test_image": test_image,
        "similarity": similarity,
        "threshold": MATCH_THRESHOLD,
        "match": matched
    }


def main():

    try:

        if len(sys.argv) < 3:

            raise ValueError(
                "Usage: python person_matcher.py "
                "<reference_image> <test_image>"
            )

        result = compare(
            sys.argv[1],
            sys.argv[2]
        )

        print(
            json.dumps(
                {
                    "success": True,
                    "result": result
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