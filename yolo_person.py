import sys
import json
import cv2
import numpy as np
import onnxruntime as ort
import os


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "yolo26n.onnx"
)

IMAGE_SIZE = 640

PERSON_CLASS_ID = 0
CONFIDENCE_THRESHOLD = 0.25


def load_model():

    return ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )


def preprocess(image):

    image_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image_resized = cv2.resize(
        image_rgb,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    tensor = image_resized.astype(
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


def detect_people(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    height, width = image.shape[:2]

    session = load_model()

    input_name = session.get_inputs()[0].name

    tensor = preprocess(image)

    outputs = session.run(
        None,
        {
            input_name: tensor
        }
    )

    predictions = outputs[0]

    # Remove batch dimension
    predictions = predictions[0]

    detections = []

    for prediction in predictions:

        x1, y1, x2, y2, confidence, class_id = prediction

        confidence = float(confidence)
        class_id = int(class_id)

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        if class_id != PERSON_CLASS_ID:
            continue

        # Convert model coordinates back to original image size
        scale_x = width / IMAGE_SIZE
        scale_y = height / IMAGE_SIZE

        x1 = int(x1 * scale_x)
        y1 = int(y1 * scale_y)
        x2 = int(x2 * scale_x)
        y2 = int(y2 * scale_y)

        detections.append(
            {
                "class": "person",
                "class_id": class_id,
                "confidence": confidence,
                "box": [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            }
        )

    return {
        "image": image_path,
        "image_width": width,
        "image_height": height,
        "persons_detected": len(detections),
        "detections": detections
    }


def main():

    try:

        if len(sys.argv) < 2:

            raise ValueError(
                "Usage: python yolo_person.py <image_path>"
            )

        result = detect_people(
            sys.argv[1]
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