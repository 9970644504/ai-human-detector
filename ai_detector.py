import sys
import json
import os

import cv2
import numpy as np
import onnxruntime as ort


MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "models",
    "yolo26n.onnx"
)

IMAGE_SIZE = 640
CONFIDENCE_THRESHOLD = 0.40


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )


def preprocess(image):
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    image = image.astype(
        np.float32
    ) / 255.0

    image = np.transpose(
        image,
        (2, 0, 1)
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


def detect_image(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            "OpenCV could not read the image."
        )

    height, width = image.shape[:2]

    session = load_model()

    input_name = session.get_inputs()[0].name

    input_tensor = preprocess(image)

    outputs = session.run(
        None,
        {
            input_name: input_tensor
        }
    )

    predictions = outputs[0]

    detections = []

    # YOLO output is expected to be:
    # [1, 300, 6]
    #
    # Each detection:
    # [x1, y1, x2, y2, confidence, class_id]

    for detection in predictions[0]:

        x1, y1, x2, y2, confidence, class_id = detection

        confidence = float(confidence)
        class_id = int(class_id)

        # COCO class 0 = person
        if class_id != 0:
            continue

        if confidence < CONFIDENCE_THRESHOLD:
            continue

        # Convert 640x640 coordinates back
        # to the original image size.

        x1 = int(x1 * width / IMAGE_SIZE)
        y1 = int(y1 * height / IMAGE_SIZE)

        x2 = int(x2 * width / IMAGE_SIZE)
        y2 = int(y2 * height / IMAGE_SIZE)

        detections.append({
            "class": "person",
            "confidence": round(confidence, 4),
            "box": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2
            }
        })

    return {
        "image": image_path,
        "width": width,
        "height": height,
        "people_detected": len(detections),
        "detections": detections
    }


def main():

    try:

        if len(sys.argv) < 2:
            raise ValueError(
                "Usage: python ai_detector.py <image_path>"
            )

        image_path = sys.argv[1]

        result = detect_image(image_path)

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