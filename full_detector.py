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

PERSON_CLASS_ID = 0
PERSON_CONFIDENCE = 0.25

MATCH_THRESHOLD = 0.80


def load_yolo():

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


def preprocess(image):

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


def detect_person(yolo, image):

    height, width = image.shape[:2]

    input_name = yolo.get_inputs()[0].name

    tensor = preprocess(image)

    outputs = yolo.run(
        None,
        {
            input_name: tensor
        }
    )

    predictions = outputs[0][0]

    persons = []

    for prediction in predictions:

        x1, y1, x2, y2, confidence, class_id = prediction

        confidence = float(confidence)
        class_id = int(class_id)

        if confidence < PERSON_CONFIDENCE:
            continue

        if class_id != PERSON_CLASS_ID:
            continue

        scale_x = width / IMAGE_SIZE
        scale_y = height / IMAGE_SIZE

        persons.append(
            {
                "confidence": confidence,
                "box": [
                    int(x1 * scale_x),
                    int(y1 * scale_y),
                    int(x2 * scale_x),
                    int(y2 * scale_y)
                ]
            }
        )

    return persons


def get_face_embedding(face_app, image):

    faces = face_app.get(image)

    if len(faces) == 0:
        return None

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
            * np.linalg.norm(b)
        )
    )


def main():

    try:

        if len(sys.argv) < 3:

            raise ValueError(
                "Usage: python full_detector.py "
                "<reference_image> <test_image>"
            )

        reference_path = sys.argv[1]
        test_path = sys.argv[2]

        reference = cv2.imread(
            reference_path
        )

        test = cv2.imread(
            test_path
        )

        if reference is None:
            raise ValueError(
                "Could not read reference image."
            )

        if test is None:
            raise ValueError(
                "Could not read test image."
            )

        yolo = load_yolo()

        face_app = load_face_model()

        # Reference face
        reference_embedding = get_face_embedding(
            face_app,
            reference
        )

        if reference_embedding is None:

            raise ValueError(
                "No face detected in reference image."
            )

        # YOLO person detection
        persons = detect_person(
            yolo,
            test
        )

        if len(persons) == 0:

            result = {
                "success": True,
                "person_detected": False,
                "face_detected": False,
                "match": False
            }

            print(
                json.dumps(
                    result,
                    indent=2
                )
            )

            return

        # InsightFace
        test_embedding = get_face_embedding(
            face_app,
            test
        )

        if test_embedding is None:

            result = {
                "success": True,
                "person_detected": True,
                "face_detected": False,
                "match": False
            }

            print(
                json.dumps(
                    result,
                    indent=2
                )
            )

            return

        similarity = cosine_similarity(
            reference_embedding,
            test_embedding
        )

        result = {
            "success": True,
            "person_detected": True,
            "person_count": len(persons),
            "face_detected": True,
            "similarity": similarity,
            "threshold": MATCH_THRESHOLD,
            "match": similarity >= MATCH_THRESHOLD,
            "persons": persons
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
