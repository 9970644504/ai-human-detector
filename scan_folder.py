import sys
import json
import os

import cv2
import numpy as np
from insightface.app import FaceAnalysis


MATCH_THRESHOLD = 0.80

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

OUTPUT_FILE = "scan_results.json"


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
        return None

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


def cosine_similarity(a, b):

    return float(
        np.dot(a, b)
        /
        (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )
    )


def scan_folder(reference_image, folder):

    if not os.path.isfile(reference_image):
        raise FileNotFoundError(
            f"Reference image not found: {reference_image}"
        )

    if not os.path.isdir(folder):
        raise NotADirectoryError(
            f"Folder not found: {folder}"
        )

    app = load_model()

    reference_embedding = get_embedding(
        app,
        reference_image
    )

    if reference_embedding is None:
        raise ValueError(
            "No face detected in reference image."
        )

    results = []

    for root, _, files in os.walk(folder):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension not in IMAGE_EXTENSIONS:
                continue

            image_path = os.path.join(
                root,
                filename
            )

            if os.path.abspath(image_path) == os.path.abspath(
                reference_image
            ):
                continue

            try:

                embedding = get_embedding(
                    app,
                    image_path
                )

                if embedding is None:

                    results.append(
                        {
                            "image": image_path,
                            "status": "NO_FACE"
                        }
                    )

                    continue

                similarity = cosine_similarity(
                    reference_embedding,
                    embedding
                )

                results.append(
                    {
                        "image": image_path,
                        "similarity": similarity,
                        "match": (
                            similarity >= MATCH_THRESHOLD
                        )
                    }
                )

            except Exception as error:

                results.append(
                    {
                        "image": image_path,
                        "status": "ERROR",
                        "error": str(error)
                    }
                )

    return results


def main():

    try:

        if len(sys.argv) < 3:

            raise ValueError(
                "Usage: python scan_folder.py "
                "<reference_image> <folder>"
            )

        reference_image = sys.argv[1]
        folder = sys.argv[2]

        results = scan_folder(
            reference_image,
            folder
        )

        matches = [
            result
            for result in results
            if result.get("match") is True
        ]

        output = {
            "success": True,
            "reference_image": reference_image,
            "folder": folder,
            "threshold": MATCH_THRESHOLD,
            "images_scanned": len(results),
            "matches_found": len(matches),
            "results": results
        }

        with open(
            OUTPUT_FILE,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                output,
                file,
                indent=2
            )

        print(
            json.dumps(
                output,
                indent=2
            )
        )

        print(
            f"\nResults saved to: {OUTPUT_FILE}"
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