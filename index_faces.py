import os
import sys
import json
import time
import hashlib
from pathlib import Path

import cv2
import numpy as np

from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif"
}

INDEX_VERSION = 1

INDEX_FILENAME = "face_index.json"

MODEL_NAME = "buffalo_l"

DET_SIZE = (640, 640)


# ============================================================
# LOGGING
# ============================================================

def log(message):
    """
    Send normal progress information to Electron.
    """
    print(message, flush=True)


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

def load_face_model():

    log("Loading InsightFace...")

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=[
            "CPUExecutionProvider"
        ]
    )

    app.prepare(
        ctx_id=0,
        det_size=DET_SIZE
    )

    log("InsightFace loaded.")

    return app


# ============================================================
# IMAGE CHECK
# ============================================================

def is_image_file(path):

    return (
        Path(path)
        .suffix
        .lower()
        in IMAGE_EXTENSIONS
    )


# ============================================================
# FIND IMAGES
# ============================================================

def get_image_files(folder):

    files = []

    folder = os.path.abspath(folder)

    if not os.path.isdir(folder):

        return files

    for root, _, filenames in os.walk(folder):

        for filename in filenames:

            path = os.path.join(
                root,
                filename
            )

            if is_image_file(path):

                files.append(
                    os.path.abspath(path)
                )

    return sorted(files)


# ============================================================
# FILE SIGNATURE
# ============================================================

def get_file_signature(path):

    try:

        stat = os.stat(path)

        return {
            "size": stat.st_size,
            "modified": stat.st_mtime
        }

    except Exception:

        return {
            "size": 0,
            "modified": 0
        }


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(embedding):

    embedding = np.asarray(
        embedding,
        dtype=np.float32
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm == 0:

        return embedding

    return embedding / norm


# ============================================================
# CREATE INDEX ENTRY
# ============================================================

def process_image(
    app,
    image_path
):

    record = {

        "path": os.path.abspath(
            image_path
        ),

        "signature":
            get_file_signature(
                image_path
            ),

        "faces": [],

        "face_count": 0,

        "error": None
    }

    try:

        image = cv2.imread(
            image_path
        )

        if image is None:

            record["error"] = (
                "Could not read image."
            )

            return record

        faces = app.get(
            image
        )

        record["face_count"] = len(
            faces
        )

        for face in faces:

            embedding = normalize_embedding(
                face.embedding
            )

            record["faces"].append(
                embedding.tolist()
            )

        return record

    except Exception as error:

        record["error"] = str(
            error
        )

        return record


# ============================================================
# LOAD EXISTING INDEX
# ============================================================

def load_existing_index(
    index_path
):

    if not os.path.exists(
        index_path
    ):

        return {
            "version": INDEX_VERSION,
            "created": time.time(),
            "updated": time.time(),
            "folder": None,
            "images": []
        }

    try:

        with open(
            index_path,
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(
                file
            )

        if not isinstance(
            data,
            dict
        ):

            raise ValueError(
                "Invalid index format."
            )

        return data

    except Exception as error:

        log(
            f"Existing index could not be loaded: {error}"
        )

        return {
            "version": INDEX_VERSION,
            "created": time.time(),
            "updated": time.time(),
            "folder": None,
            "images": []
        }


# ============================================================
# SAVE INDEX
# ============================================================

def save_index(
    index,
    index_path
):

    temporary_path = (
        index_path + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            index,
            file,
            ensure_ascii=False,
            separators=(
                ",",
                ":"
            )
        )

    os.replace(
        temporary_path,
        index_path
    )


# ============================================================
# MAIN INDEX BUILDER
# ============================================================

def build_index(folder):

    folder = os.path.abspath(
        folder
    )

    if not os.path.isdir(folder):

        raise ValueError(
            "Selected folder does not exist."
        )

    index_path = os.path.join(
        folder,
        INDEX_FILENAME
    )

    log("")
    log("=" * 60)
    log("FACE INDEX BUILDER")
    log("=" * 60)
    log("")

    log(
        f"Folder: {folder}"
    )

    # --------------------------------------------------------
    # FIND PHOTOS
    # --------------------------------------------------------

    log("Finding photos...")

    image_files = get_image_files(
        folder
    )

    total = len(
        image_files
    )

    log(
        f"Found {total} image(s)."
    )

    if total == 0:

        raise ValueError(
            "No supported images were found."
        )

    # --------------------------------------------------------
    # LOAD OLD INDEX
    # --------------------------------------------------------

    index = load_existing_index(
        index_path
    )

    index["version"] = INDEX_VERSION
    index["folder"] = folder
    index["updated"] = time.time()

    if "images" not in index:

        index["images"] = []

    # --------------------------------------------------------
    # OLD RECORD LOOKUP
    # --------------------------------------------------------

    existing = {}

    for record in index["images"]:

        path = record.get(
            "path"
        )

        if path:

            existing[
                os.path.abspath(path)
            ] = record

    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    app = load_face_model()

    log("")

    # --------------------------------------------------------
    # STATISTICS
    # --------------------------------------------------------

    processed = 0
    skipped = 0
    failed = 0
    faces_found = 0

    new_records = []

    start_time = time.time()

    # --------------------------------------------------------
    # PROCESS PHOTOS
    # --------------------------------------------------------

    for number, image_path in enumerate(
        image_files,
        start=1
    ):

        current_signature = (
            get_file_signature(
                image_path
            )
        )

        old_record = existing.get(
            os.path.abspath(
                image_path
            )
        )

        # ----------------------------------------------------
        # SKIP UNCHANGED IMAGE
        # ----------------------------------------------------

        if (
            old_record
            and
            old_record.get(
                "signature"
            ) == current_signature
        ):

            new_records.append(
                old_record
            )

            skipped += 1

            percent = (
                number / total
            ) * 100

            log(
                f"INDEX_PROGRESS|"
                f"{number}|"
                f"{total}|"
                f"{percent:.1f}|"
                f"Skipped: {os.path.basename(image_path)}"
            )

            continue

        # ----------------------------------------------------
        # PROCESS IMAGE
        # ----------------------------------------------------

        record = process_image(
            app,
            image_path
        )

        new_records.append(
            record
        )

        processed += 1

        if record["error"]:

            failed += 1

        faces_found += record[
            "face_count"
        ]

        percent = (
            number / total
        ) * 100

        log(
            f"INDEX_PROGRESS|"
            f"{number}|"
            f"{total}|"
            f"{percent:.1f}|"
            f"Scanning: {os.path.basename(image_path)}"
        )

        # ----------------------------------------------------
        # SAVE PERIODICALLY
        # ----------------------------------------------------

        if (
            number % 25 == 0
            or
            number == total
        ):

            index["images"] = (
                new_records
            )

            index["updated"] = (
                time.time()
            )

            save_index(
                index,
                index_path
            )

    # --------------------------------------------------------
    # FINAL SAVE
    # --------------------------------------------------------

    index["images"] = (
        new_records
    )

    index["updated"] = (
        time.time()
    )

    index["statistics"] = {

        "total_images": total,

        "processed_images": processed,

        "skipped_images": skipped,

        "failed_images": failed,

        "faces_found": faces_found
    }

    save_index(
        index,
        index_path
    )

    elapsed = (
        time.time()
        -
        start_time
    )

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------

    log("")

    log("=" * 60)
    log("INDEX COMPLETE")
    log("=" * 60)

    log(
        f"Images:       {total}"
    )

    log(
        f"Processed:    {processed}"
    )

    log(
        f"Skipped:      {skipped}"
    )

    log(
        f"Failed:       {failed}"
    )

    log(
        f"Faces found:  {faces_found}"
    )

    log(
        f"Time:         {elapsed:.2f} seconds"
    )

    log(
        f"Index:        {index_path}"
    )

    log("=" * 60)

    return index


# ============================================================
# COMMAND LINE
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            "Usage:"
        )

        print(
            "python index_faces.py <folder>"
        )

        sys.exit(1)

    folder = sys.argv[1]

    try:

        build_index(
            folder
        )

    except Exception as error:

        log(
            f"INDEX_ERROR|{error}"
        )

        sys.exit(1)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()