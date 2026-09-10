import os
import cv2
import numpy as np

from insightface.app import FaceAnalysis


REFERENCE_IMAGE = r"C:\Users\dell\Pictures\person.jpg.jpg"

SAME_FOLDER = r".\calibration\same"
DIFFERENT_FOLDER = r".\calibration\different"


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}


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
        return []

    faces = app.get(image)

    return [
        face.embedding
        for face in faces
    ]


def cosine_similarity(a, b):

    denominator = (
        np.linalg.norm(a)
        *
        np.linalg.norm(b)
    )

    if denominator == 0:
        return 0.0

    return float(
        np.dot(a, b) / denominator
    )


def get_images(folder):

    images = []

    for root, _, files in os.walk(folder):

        for filename in files:

            extension = os.path.splitext(
                filename
            )[1].lower()

            if extension in IMAGE_EXTENSIONS:

                images.append(
                    os.path.abspath(
                        os.path.join(
                            root,
                            filename
                        )
                    )
                )

    return images


def calculate_scores(
    app,
    reference_embedding,
    folder
):

    images = get_images(folder)

    scores = []

    for image_path in images:

        embeddings = get_embeddings(
            app,
            image_path
        )

        if not embeddings:

            print(
                f"NO FACE: {image_path}"
            )

            continue

        similarities = [
            cosine_similarity(
                reference_embedding,
                embedding
            )
            for embedding in embeddings
        ]

        best = max(similarities)

        scores.append(best)

        print(
            f"{best:.4f}  {image_path}"
        )

    return scores


def main():

    print()
    print("=" * 60)
    print("FACE THRESHOLD CALIBRATION")
    print("=" * 60)

    app = load_model()

    reference_embeddings = get_embeddings(
        app,
        REFERENCE_IMAGE
    )

    if not reference_embeddings:

        raise ValueError(
            "No face found in reference image."
        )

    reference_embedding = (
        reference_embeddings[0]
    )

    print()
    print("SAME PERSON")
    print("-" * 60)

    same_scores = calculate_scores(
        app,
        reference_embedding,
        SAME_FOLDER
    )

    print()
    print("DIFFERENT PEOPLE")
    print("-" * 60)

    different_scores = calculate_scores(
        app,
        reference_embedding,
        DIFFERENT_FOLDER
    )

    print()
    print("=" * 60)
    print("CALIBRATION RESULTS")
    print("=" * 60)

    if same_scores:

        print()
        print("Same-person scores:")

        print(
            [
                round(score, 4)
                for score in same_scores
            ]
        )

        print(
            f"Minimum same-person: "
            f"{min(same_scores):.4f}"
        )

        print(
            f"Maximum same-person: "
            f"{max(same_scores):.4f}"
        )

        print(
            f"Average same-person: "
            f"{np.mean(same_scores):.4f}"
        )

    if different_scores:

        print()
        print("Different-person scores:")

        print(
            [
                round(score, 4)
                for score in different_scores
            ]
        )

        print(
            f"Minimum different-person: "
            f"{min(different_scores):.4f}"
        )

        print(
            f"Maximum different-person: "
            f"{max(different_scores):.4f}"
        )

        print(
            f"Average different-person: "
            f"{np.mean(different_scores):.4f}"
        )

    if same_scores and different_scores:

        same_min = min(same_scores)
        different_max = max(different_scores)

        print()
        print("=" * 60)

        if same_min > different_max:

            suggested = (
                same_min + different_max
            ) / 2

            print(
                "GOOD SEPARATION FOUND"
            )

            print(
                f"Suggested threshold: "
                f"{suggested:.4f}"
            )

        else:

            print(
                "OVERLAP DETECTED"
            )

            print(
                "The same-person and "
                "different-person scores overlap."
            )

            print(
                "We need more calibration images "
                "before choosing a reliable threshold."
            )

        print("=" * 60)


if __name__ == "__main__":
    main()