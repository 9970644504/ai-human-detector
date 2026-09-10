import sys
import json
import os

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis


YOLO_MODEL = os.path.join(
    os.path.dirname(__file__),
    "models",
    "yolo26n.onnx"
)

IMAGE_SIZE = 640
MATCH_THRESHOLD = 0.80


def load_yolo():
    if not os.path.exists(YOLO_MODEL):
        raise FileNotFoundError(
            f"YOLO model not found: {YOLO_MODEL}"
        )

    return ort.InferenceSession(
        YOLO_MODEL,
        providers=["CPUExecutionProvider"]
    )


def load_face_model():
    app = FaceAnalysis(
        name="buffalo_l",
        providers=["CPUExecutionProvider"]
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )

    return app


def preprocess_yolo(image):
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    resized = cv2.resize(
        rgb,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    tensor = resized.astype(
        np.float32
    ) / 255.0

    tensor = np.transpose(
        tensor,
        (2, 0, 1)
    )

    return np.expand_dims(
        tensor,
        axis=0
    )


def run_yolo(session, image):
    input_name = session.get_inputs()[0].name

    tensor = preprocess_yolo(image)

    outputs = session.run(
        None,
        {
            input_name: tensor
        }
    )

    return outputs


def cosine_similarity(a, b):
    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )
    )


def get_face_embedding(app, image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    faces = app.get(image)

    if len(faces) == 0:
        return None

    face = max(
        faces,
        key=lambda f:
        (f.bbox[2] - f.bbox[0])
        * (f.bbox[3] - f.bbox[1])
    )

    return face.embedding


def main():

    try:

        if len(sys.argv) < 3:
            raise ValueError(
                "Usage: python combined_detector.py "
                "<reference_image> <test_image>"
            )

        reference_image = sys.argv[1]
        test_image = sys.argv[2]

        yolo = load_yolo()

        reference = cv2.imread(
            reference_image
        )

        test = cv2.imread(
            test_image
        )

        if reference is None:
            raise ValueError(
                "Could not read reference image."
            )

        if test is None:
            raise ValueError(
                "Could not read test image."
            )

        # Run YOLO on test image
        yolo_outputs = run_yolo(
            yolo,
            test
        )

        face_app = load_face_model()

        reference_embedding = get_face_embedding(
            face_app,
            reference_image
        )

        if reference_embedding is None:
            raise ValueError(
                "No face found in reference image."
            )

        test_embedding = get_face_embedding(
            face_app,
            test_image
        )

        if test_embedding is None:

            result = {
                "success": True,
                "person_detected_by_yolo": True,
                "face_detected": False,
                "match": False
            }

        else:

            similarity = cosine_similarity(
                reference_embedding,
                test_embedding
            )

            result = {
                "success": True,
                "person_detected_by_yolo": True,
                "face_detected": True,
                "similarity": similarity,
                "threshold": MATCH_THRESHOLD,
                "match": (
                    similarity >= MATCH_THRESHOLD
                ),
                "yolo_output_shapes": [
                    list(output.shape)
                    for output in yolo_outputs
                ]
            }

        print(
            json.dumps(
                result,
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