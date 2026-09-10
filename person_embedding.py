import sys
import json
import cv2
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


def get_embedding(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    app = load_model()

    faces = app.get(image)

    if len(faces) == 0:
        raise ValueError(
            "No face detected in the image."
        )

    # Use the largest detected face
    face = max(
        faces,
        key=lambda f: (f.bbox[2] - f.bbox[0])
        * (f.bbox[3] - f.bbox[1])
    )

    embedding = face.embedding

    return {
        "image": image_path,
        "faces_detected": len(faces),
        "embedding_length": len(embedding),
        "embedding_norm": float(
            (embedding ** 2).sum() ** 0.5
        )
    }


def main():

    try:

        if len(sys.argv) < 2:
            raise ValueError(
                "Usage: python person_embedding.py <image_path>"
            )

        result = get_embedding(sys.argv[1])

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
