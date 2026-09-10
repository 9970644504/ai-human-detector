import os
import sys
import json
import traceback
from pathlib import Path

import cv2
import numpy as np

from ultralytics import YOLO
from insightface.app import FaceAnalysis


# ============================================================
# SETTINGS
# ============================================================

MATCH_THRESHOLD = 0.40
UNCERTAIN_THRESHOLD = 0.30

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif"
}


# ============================================================
# PROGRESS
# ============================================================

def send_progress(
    current,
    total,
    message,
    status="scanning"
):

    if total > 0:
        percent = int(
            (current / total) * 100
        )
    else:
        percent = 0


    data = {
        "type": "progress",
        "current": current,
        "total": total,
        "percent": percent,
        "message": message,
        "status": status
    }


    print(
        "PROGRESS_JSON:" +
        json.dumps(
            data,
            ensure_ascii=False
        ),
        flush=True
    )


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

def load_face_model():

    print(
        "Loading InsightFace...",
        flush=True
    )


    app = FaceAnalysis(
        name="buffalo_l",
        providers=[
            "CPUExecutionProvider"
        ]
    )


    app.prepare(
        ctx_id=0,
        det_size=(640, 640)
    )


    return app


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    a,
    b
):

    a = np.asarray(
        a,
        dtype=np.float32
    )

    b = np.asarray(
        b,
        dtype=np.float32
    )


    denominator = (
        np.linalg.norm(a) *
        np.linalg.norm(b)
    )


    if denominator == 0:

        return 0.0


    return float(
        np.dot(a, b) /
        denominator
    )


# ============================================================
# REFERENCE EMBEDDING
# ============================================================

def get_reference_embedding(
    app,
    image_path
):

    image = cv2.imread(
        image_path
    )


    if image is None:

        raise ValueError(
            "Could not read reference image."
        )


    faces = app.get(
        image
    )


    if not faces:

        raise ValueError(
            "No face found in reference image."
        )


    # Choose largest face as reference
    face = max(
        faces,
        key=lambda f:
            (
                f.bbox[2] - f.bbox[0]
            ) *
            (
                f.bbox[3] - f.bbox[1]
            )
    )


    return face.embedding


# ============================================================
# FIND IMAGES
# ============================================================

def image_files(
    folder
):

    files = []


    for root, _, names in os.walk(
        folder
    ):

        for name in names:

            extension = (
                Path(name)
                .suffix
                .lower()
            )


            if extension in IMAGE_EXTENSIONS:

                files.append(
                    os.path.join(
                        root,
                        name
                    )
                )


    return sorted(
        files,
        key=lambda x:
            x.lower()
    )


# ============================================================
# CREATE RESULT RECORD
# ============================================================

def create_record(
    image_path
):

    return {

        "image":
            os.path.abspath(
                image_path
            ),

        "status":
            None,

        "best_similarity":
            None,

        "faces_detected":
            0,

        "person_detected":
            False,

        "error":
            None

    }


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # CHECK ARGUMENTS
    # --------------------------------------------------------

    if len(sys.argv) != 3:

        print(
            "Usage: python full_scan.py "
            "<reference_image> <folder>",
            file=sys.stderr,
            flush=True
        )

        sys.exit(1)


    reference_path = (
        os.path.abspath(
            sys.argv[1]
        )
    )


    folder = (
        os.path.abspath(
            sys.argv[2]
        )
    )


    # --------------------------------------------------------
    # CHECK REFERENCE
    # --------------------------------------------------------

    if not os.path.isfile(
        reference_path
    ):

        raise FileNotFoundError(
            "Reference image was not found: "
            + reference_path
        )


    # --------------------------------------------------------
    # CHECK FOLDER
    # --------------------------------------------------------

    if not os.path.isdir(
        folder
    ):

        raise NotADirectoryError(
            "Photo folder was not found: "
            + folder
        )


    # --------------------------------------------------------
    # FIND IMAGES
    # --------------------------------------------------------

    images = image_files(
        folder
    )


    if not images:

        raise RuntimeError(
            "No supported images were found."
        )


    total = len(
        images
    )


    print(
        f"Found {total} image(s).",
        flush=True
    )


    send_progress(
        0,
        total,
        f"Found {total} image(s)."
    )


    # --------------------------------------------------------
    # LOAD YOLO
    # --------------------------------------------------------

    print(
        "Loading YOLO...",
        flush=True
    )


    try:

        yolo = YOLO(
            "yolov8n.pt"
        )

    except Exception as error:

        raise RuntimeError(
            "Could not load YOLO model: "
            + str(error)
        )


    # --------------------------------------------------------
    # LOAD INSIGHTFACE
    # --------------------------------------------------------

    app = load_face_model()


    # --------------------------------------------------------
    # REFERENCE EMBEDDING
    # --------------------------------------------------------

    print(
        "Creating reference embedding...",
        flush=True
    )


    reference = (
        get_reference_embedding(
            app,
            reference_path
        )
    )


    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = {

        "images_found":
            total,

        "images_processed":
            0,

        "matches":
            0,

        "uncertain":
            0,

        "no_matches":
            0,

        "no_faces":
            0,

        "no_person":
            0,

        "errors":
            0

    }


    results = []


    # ========================================================
    # SCAN EACH IMAGE
    # ========================================================

    for index, image_path in enumerate(
        images,
        start=1
    ):

        filename = os.path.basename(
            image_path
        )


        send_progress(
            index - 1,
            total,
            f"Scanning {filename}..."
        )


        record = create_record(
            image_path
        )


        try:

            # ------------------------------------------------
            # READ IMAGE
            # ------------------------------------------------

            image = cv2.imread(
                image_path
            )


            if image is None:

                record["status"] = "ERROR"

                record["error"] = (
                    "Could not read image."
                )

                summary["errors"] += 1

                results.append(
                    record
                )

                send_progress(
                    index,
                    total,
                    f"Could not read {filename}."
                )

                continue


            # ------------------------------------------------
            # YOLO PERSON DETECTION
            # ------------------------------------------------

            yolo_results = yolo(
                image,
                verbose=False
            )


            person_found = False


            if yolo_results:

                yolo_result = (
                    yolo_results[0]
                )


                if (
                    yolo_result.boxes
                    is not None
                ):

                    classes = (
                        yolo_result
                        .boxes
                        .cls
                        .tolist()
                    )


                    for cls in classes:

                        # COCO class 0 = person
                        if int(cls) == 0:

                            person_found = True

                            break


            record[
                "person_detected"
            ] = person_found


            # ------------------------------------------------
            # NO PERSON
            # ------------------------------------------------

            if not person_found:

                record[
                    "status"
                ] = "NO_PERSON"


                summary[
                    "no_person"
                ] += 1


                results.append(
                    record
                )


                send_progress(
                    index,
                    total,
                    f"No person found: {filename}"
                )


                continue


            # ------------------------------------------------
            # INSIGHTFACE
            # ------------------------------------------------

            faces = app.get(
                image
            )


            record[
                "faces_detected"
            ] = len(faces)


            # ------------------------------------------------
            # NO FACE
            # ------------------------------------------------

            if not faces:

                record[
                    "status"
                ] = "NO_FACE"


                summary[
                    "no_faces"
                ] += 1


                results.append(
                    record
                )


                send_progress(
                    index,
                    total,
                    f"No face found: {filename}"
                )


                continue


            # ------------------------------------------------
            # COMPARE FACES
            # ------------------------------------------------

            scores = []


            for face in faces:

                score = cosine_similarity(
                    reference,
                    face.embedding
                )


                scores.append(
                    score
                )


            best = max(
                scores
            )


            record[
                "best_similarity"
            ] = round(
                best,
                6
            )


            # ------------------------------------------------
            # MATCH
            # ------------------------------------------------

            if best >= MATCH_THRESHOLD:

                record[
                    "status"
                ] = "MATCH"


                summary[
                    "matches"
                ] += 1


            # ------------------------------------------------
            # UNCERTAIN
            # ------------------------------------------------

            elif best >= UNCERTAIN_THRESHOLD:

                record[
                    "status"
                ] = "UNCERTAIN"


                summary[
                    "uncertain"
                ] += 1


            # ------------------------------------------------
            # NO MATCH
            # ------------------------------------------------

            else:

                record[
                    "status"
                ] = "NO_MATCH"


                summary[
                    "no_matches"
                ] += 1


            summary[
                "images_processed"
            ] += 1


            results.append(
                record
            )


            score_text = (
                f"{best:.3f}"
            )


            send_progress(
                index,
                total,
                f"{record['status']}: "
                f"{filename} "
                f"({score_text})"
            )


        except Exception as error:

            record[
                "status"
            ] = "ERROR"


            record[
                "error"
            ] = str(error)


            summary[
                "errors"
            ] += 1


            results.append(
                record
            )


            send_progress(
                index,
                total,
                f"Error scanning {filename}: "
                f"{error}"
            )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    output = {

        "reference_image":
            reference_path,

        "scan_folder":
            folder,

        "match_threshold":
            MATCH_THRESHOLD,

        "uncertain_threshold":
            UNCERTAIN_THRESHOLD,

        "summary":
            summary,

        "results":
            results

    }


    # --------------------------------------------------------
    # SAVE JSON FILE
    # --------------------------------------------------------

    result_file = os.path.join(
        os.path.dirname(
            os.path.abspath(
                __file__
            )
        ),
        "full_scan_results.json"
    )


    with open(
        result_file,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )


    # --------------------------------------------------------
    # COMPLETE PROGRESS
    # --------------------------------------------------------

    send_progress(
        total,
        total,
        "Scan complete.",
        "complete"
    )


    print(
        "",
        flush=True
    )


    print(
        "=" * 60,
        flush=True
    )


    print(
        "SCAN COMPLETE",
        flush=True
    )


    print(
        "=" * 60,
        flush=True
    )


    for key, value in summary.items():

        print(
            f"{key.replace('_', ' ').title():22} "
            f"{value}",
            flush=True
        )


    print(
        f"Match threshold:       {MATCH_THRESHOLD}",
        flush=True
    )


    print(
        f"Uncertain threshold:   {UNCERTAIN_THRESHOLD}",
        flush=True
    )


    print(
        f"Results saved to:      {result_file}",
        flush=True
    )


    print(
        "=" * 60,
        flush=True
    )


    # --------------------------------------------------------
    # IMPORTANT:
    # ELECTRON READS THIS LINE
    # --------------------------------------------------------

    print(
        "RESULT_JSON:" +
        json.dumps(
            output,
            ensure_ascii=False
        ),
        flush=True
    )


# ============================================================
# ERROR HANDLER
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        print(
            "",
            file=sys.stderr,
            flush=True
        )


        print(
            "SCAN ERROR:",
            file=sys.stderr,
            flush=True
        )


        print(
            str(error),
            file=sys.stderr,
            flush=True
        )


        traceback.print_exc(
            file=sys.stderr
        )


        sys.exit(1)