# ============================================================
# AI PERSON PHOTO FINDER
# SEARCH INDEX
#
# Supports face_index.json VERSION 1 and VERSION 2
#
# INPUT:
#   python search_index.py "<reference_image>" "<face_index.json>"
#
# OUTPUT:
#   PROGRESS_JSON:{...}
#   RESULT_JSON:{...}
# ============================================================

import sys
import json
import math
from pathlib import Path

import cv2
import numpy as np
from insightface.app import FaceAnalysis


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_NAME = "buffalo_l"

# ------------------------------------------------------------
# IMPORTANT:
#
# Older index files may have version 1.
# Your current index appears to be version 2.
#
# Therefore we support BOTH.
# ------------------------------------------------------------

SUPPORTED_INDEX_VERSIONS = {1, 2}

EXPECTED_EMBEDDING_DIMENSIONS = 512

MATCH_THRESHOLD = 0.40
UNCERTAIN_THRESHOLD = 0.30

SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}


# ============================================================
# PROGRESS
# ============================================================

def send_progress(
    message,
    percent=None,
    current=None,
    total=None,
    progress_type="searching",
):
    data = {
        "type": progress_type,
        "message": str(message),
    }

    if percent is not None:
        data["percent"] = float(percent)

    if current is not None:
        data["current"] = int(current)

    if total is not None:
        data["total"] = int(total)

    print(
        "PROGRESS_JSON:" +
        json.dumps(
            data,
            ensure_ascii=False,
        ),
        flush=True,
    )


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(image_path):
    """
    Safely load an image.

    np.fromfile + cv2.imdecode works better with
    Windows paths containing Unicode characters.
    """

    image_path = Path(image_path)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image does not exist: {image_path}"
        )

    if not image_path.is_file():
        raise ValueError(
            f"Image path is not a file: {image_path}"
        )

    try:
        data = np.fromfile(
            str(image_path),
            dtype=np.uint8,
        )

        if data.size == 0:
            raise ValueError(
                f"Image file is empty: {image_path}"
            )

        image = cv2.imdecode(
            data,
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise ValueError(
                f"OpenCV could not decode image: {image_path}"
            )

        return image

    except Exception as error:
        raise ValueError(
            f"Could not load image '{image_path}': {error}"
        )


# ============================================================
# MODEL
# ============================================================

def load_model():

    send_progress(
        "Loading AI face model...",
        5,
        progress_type="model-loading",
    )

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=[
            "CPUExecutionProvider"
        ],
    )

    app.prepare(
        ctx_id=0,
        det_size=(640, 640),
    )

    return app


# ============================================================
# NORMALIZE EMBEDDING
# ============================================================

def normalize_embedding(embedding):
    """
    Convert an embedding to float32 and L2-normalize it.
    """

    if embedding is None:
        return None

    try:
        vector = np.asarray(
            embedding,
            dtype=np.float32,
        ).reshape(-1)
    except Exception:
        return None

    if vector.size == 0:
        return None

    if vector.size != EXPECTED_EMBEDDING_DIMENSIONS:
        return None

    norm = float(
        np.linalg.norm(vector)
    )

    if not math.isfinite(norm) or norm <= 0:
        return None

    vector = vector / norm

    return vector


# ============================================================
# CREATE REFERENCE EMBEDDING
# ============================================================

def create_reference_embedding(
    app,
    reference_path,
):

    send_progress(
        "Analyzing reference photo...",
        10,
        progress_type="reference",
    )

    # IMPORTANT:
    #
    # InsightFace app.get() requires the actual
    # OpenCV image array.
    #
    # NOT the string path.
    #

    image = load_image(
        reference_path
    )

    faces = app.get(
        image
    )

    if faces is None:
        faces = []

    if len(faces) == 0:
        raise ValueError(
            "No face was detected in the reference photo."
        )

    # --------------------------------------------------------
    # Find the face with the largest bounding box.
    # This is useful if the reference image contains
    # multiple people.
    # --------------------------------------------------------

    best_face = None
    best_area = -1.0

    for face in faces:

        try:

            embedding = getattr(
                face,
                "embedding",
                None,
            )

            if embedding is None:
                continue

            normalized = normalize_embedding(
                embedding
            )

            if normalized is None:
                continue

            bbox = getattr(
                face,
                "bbox",
                None,
            )

            area = 0.0

            if bbox is not None:

                try:

                    x1, y1, x2, y2 = [
                        float(value)
                        for value in bbox[:4]
                    ]

                    width = max(
                        0.0,
                        x2 - x1,
                    )

                    height = max(
                        0.0,
                        y2 - y1,
                    )

                    area = width * height

                except Exception:
                    area = 0.0

            if (
                best_face is None
                or area > best_area
            ):

                best_face = normalized
                best_area = area

        except Exception:
            continue

    if best_face is None:
        raise ValueError(
            "A face was detected in the reference photo, "
            "but no usable 512-dimensional face embedding "
            "could be created."
        )

    send_progress(
        f"Reference face detected ({len(faces)} face(s)).",
        15,
        progress_type="reference-ready",
    )

    return best_face, len(faces)


# ============================================================
# LOAD FACE INDEX
# ============================================================

def load_face_index(index_path):

    send_progress(
        "Reading face index...",
        20,
        progress_type="index-loading",
    )

    index_path = Path(
        index_path
    ).expanduser().resolve()

    if not index_path.exists():
        raise FileNotFoundError(
            f"Face index does not exist: {index_path}"
        )

    if not index_path.is_file():
        raise ValueError(
            f"Face index is not a file: {index_path}"
        )

    try:

        with index_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

    except json.JSONDecodeError as error:

        raise ValueError(
            f"face_index.json is not valid JSON: {error}"
        )

    except Exception as error:

        raise ValueError(
            f"Could not read face_index.json: {error}"
        )

    if not isinstance(data, dict):
        raise ValueError(
            "face_index.json must contain a JSON object."
        )

    # ========================================================
    # VERSION CHECK
    # ========================================================

    version = data.get(
        "version"
    )

    if version is not None:

        try:

            version_number = int(
                version
            )

        except Exception:

            raise ValueError(
                f"Invalid face index version: {version}"
            )

        if version_number not in SUPPORTED_INDEX_VERSIONS:

            supported = ", ".join(
                str(v)
                for v in sorted(
                    SUPPORTED_INDEX_VERSIONS
                )
            )

            raise ValueError(
                f"Unsupported face index version: "
                f"{version_number}. "
                f"Supported versions: {supported}"
            )

    # ========================================================
    # MODEL CHECK
    # ========================================================

    model = data.get(
        "model"
    )

    if model:

        if model != MODEL_NAME:

            raise ValueError(
                f"Face index was created with model "
                f"'{model}', but this search uses "
                f"'{MODEL_NAME}'."
            )

    # ========================================================
    # IMAGE LIST
    # ========================================================

    images = data.get(
        "images"
    )

    if not isinstance(images, list):

        raise ValueError(
            "face_index.json does not contain a valid "
            "'images' array."
        )

    # ========================================================
    # EMBEDDING DIMENSIONS
    # ========================================================

    dimensions = data.get(
        "embedding_dimensions"
    )

    if dimensions is not None:

        try:

            dimensions = int(
                dimensions
            )

        except Exception:

            raise ValueError(
                "Invalid embedding_dimensions in face_index.json."
            )

        if dimensions != EXPECTED_EMBEDDING_DIMENSIONS:

            raise ValueError(
                f"Index embedding dimensions are "
                f"{dimensions}, expected "
                f"{EXPECTED_EMBEDDING_DIMENSIONS}."
            )

    return data


# ============================================================
# PATH RESOLUTION
# ============================================================

def resolve_image_path(
    stored_path,
    index_path,
):

    if stored_path is None:
        return None

    if not isinstance(
        stored_path,
        str,
    ):
        return None

    stored_path = stored_path.strip()

    if not stored_path:
        return None

    path = Path(
        stored_path
    )

    # --------------------------------------------------------
    # Absolute path
    # --------------------------------------------------------

    if path.is_absolute():
        return path

    # --------------------------------------------------------
    # Relative path
    # --------------------------------------------------------

    return (
        Path(index_path).parent /
        path
    ).resolve()


# ============================================================
# GET STORED PATH
# ============================================================

def get_record_path(record):

    if not isinstance(
        record,
        dict,
    ):
        return None

    # --------------------------------------------------------
    # Support several possible field names.
    # --------------------------------------------------------

    possible_fields = [
        "path",
        "image",
        "image_path",
        "file",
        "filepath",
        "file_path",
    ]

    for field in possible_fields:

        value = record.get(
            field
        )

        if isinstance(
            value,
            str,
        ) and value.strip():

            return value.strip()

    return None


# ============================================================
# GET FACE EMBEDDINGS
# ============================================================

def get_face_embeddings(record):

    if not isinstance(
        record,
        dict,
    ):
        return []

    # --------------------------------------------------------
    # Standard format:
    #
    # "faces": [
    #     [512 values],
    #     [512 values]
    # ]
    # --------------------------------------------------------

    faces = record.get(
        "faces"
    )

    if isinstance(
        faces,
        list,
    ):

        # Already a list of embeddings.
        if len(faces) > 0:

            return faces

    # --------------------------------------------------------
    # Alternative singular embedding.
    # --------------------------------------------------------

    embedding = record.get(
        "embedding"
    )

    if embedding is not None:

        return [
            embedding
        ]

    # --------------------------------------------------------
    # Alternative embeddings field.
    # --------------------------------------------------------

    embeddings = record.get(
        "embeddings"
    )

    if isinstance(
        embeddings,
        list,
    ):

        return embeddings

    return []


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(
    reference_embedding,
    candidate_embedding,
):

    reference = np.asarray(
        reference_embedding,
        dtype=np.float32,
    ).reshape(-1)

    candidate = np.asarray(
        candidate_embedding,
        dtype=np.float32,
    ).reshape(-1)

    if (
        reference.size !=
        EXPECTED_EMBEDDING_DIMENSIONS
    ):
        return None

    if (
        candidate.size !=
        EXPECTED_EMBEDDING_DIMENSIONS
    ):
        return None

    reference_norm = float(
        np.linalg.norm(reference)
    )

    candidate_norm = float(
        np.linalg.norm(candidate)
    )

    if (
        reference_norm <= 0
        or candidate_norm <= 0
    ):
        return None

    score = float(
        np.dot(
            reference / reference_norm,
            candidate / candidate_norm,
        )
    )

    if not math.isfinite(score):
        return None

    score = max(
        -1.0,
        min(
            1.0,
            score,
        ),
    )

    return score


# ============================================================
# DETERMINE STATUS
# ============================================================

def get_status(
    similarity,
):

    if similarity >= MATCH_THRESHOLD:
        return "MATCH"

    if similarity >= UNCERTAIN_THRESHOLD:
        return "UNCERTAIN"

    return "NO_MATCH"


# ============================================================
# CREATE RESULT RECORD
# ============================================================

def make_result_record(
    image_path,
    similarity,
    status,
):

    score = float(
        similarity
    )

    absolute_path = str(
        Path(
            image_path
        ).resolve()
    )

    filename = Path(
        absolute_path
    ).name

    return {

        # Main path
        "image": absolute_path,

        # Compatibility
        "path": absolute_path,
        "image_path": absolute_path,
        "file": absolute_path,

        # File information
        "filename": filename,
        "name": filename,

        # Scores
        "score": score,
        "similarity": score,
        "best_similarity": score,

        # Classification
        "status": status,

        # Percentage
        "confidence": round(
            max(
                0.0,
                min(
                    1.0,
                    score,
                ),
            ) * 100.0,
            2,
        ),
    }


# ============================================================
# SEARCH INDEX
# ============================================================

def search_index(
    reference_embedding,
    index_data,
    index_path,
):

    images = index_data.get(
        "images",
        [],
    )

    total = len(
        images
    )

    matches = []
    uncertain = []
    all_results = []

    images_processed = 0
    images_with_faces = 0
    no_faces = 0
    errors = 0

    # ========================================================
    # PROCESS EACH INDEXED IMAGE
    # ========================================================

    for current, record in enumerate(
        images,
        start=1,
    ):

        # ----------------------------------------------------
        # Progress
        # ----------------------------------------------------

        percent = (
            20 +
            (
                current /
                max(
                    total,
                    1,
                )
            ) * 75
        )

        stored_path = get_record_path(
            record
        )

        if isinstance(
            stored_path,
            str,
        ):

            display_name = Path(
                stored_path
            ).name

        else:

            display_name = (
                f"image {current}"
            )

        send_progress(
            f"Searching: {display_name}",
            percent,
            current=current,
            total=total,
            progress_type="searching",
        )

        images_processed += 1

        # ----------------------------------------------------
        # Validate record
        # ----------------------------------------------------

        if not isinstance(
            record,
            dict,
        ):

            errors += 1
            continue

        # ----------------------------------------------------
        # Get path
        # ----------------------------------------------------

        image_path = resolve_image_path(
            stored_path,
            index_path,
        )

        if image_path is None:

            errors += 1
            continue

        # ----------------------------------------------------
        # Get face embeddings
        # ----------------------------------------------------

        face_embeddings = get_face_embeddings(
            record
        )

        if len(
            face_embeddings
        ) == 0:

            no_faces += 1
            continue

        images_with_faces += 1

        # ----------------------------------------------------
        # Compare every face.
        #
        # The best face similarity is used.
        # ----------------------------------------------------

        best_similarity = None

        for face_embedding in face_embeddings:

            try:

                candidate = normalize_embedding(
                    face_embedding
                )

                if candidate is None:
                    continue

                similarity = cosine_similarity(
                    reference_embedding,
                    candidate,
                )

                if similarity is None:
                    continue

                if (
                    best_similarity is None
                    or similarity > best_similarity
                ):

                    best_similarity = similarity

            except Exception:

                continue

        # ----------------------------------------------------
        # No usable embedding
        # ----------------------------------------------------

        if best_similarity is None:

            errors += 1
            continue

        # ----------------------------------------------------
        # Classification
        # ----------------------------------------------------

        status = get_status(
            best_similarity
        )

        result_record = make_result_record(
            image_path,
            best_similarity,
            status,
        )

        all_results.append(
            result_record
        )

        if status == "MATCH":

            matches.append(
                result_record
            )

        elif status == "UNCERTAIN":

            uncertain.append(
                result_record
            )

    # ========================================================
    # SORT
    # ========================================================

    matches.sort(
        key=lambda item: item.get(
            "similarity",
            0.0,
        ),
        reverse=True,
    )

    uncertain.sort(
        key=lambda item: item.get(
            "similarity",
            0.0,
        ),
        reverse=True,
    )

    all_results.sort(
        key=lambda item: item.get(
            "similarity",
            0.0,
        ),
        reverse=True,
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = {

        "images_found": total,

        "images_processed":
            images_processed,

        "matches":
            len(matches),

        "matching_photos":
            len(matches),

        "uncertain":
            len(uncertain),

        "uncertain_matches":
            len(uncertain),

        "no_matches":
            max(
                0,
                images_with_faces -
                len(matches) -
                len(uncertain),
            ),

        "no_faces":
            no_faces,

        "errors":
            errors,

        "images_with_faces":
            images_with_faces,

        "match_threshold":
            MATCH_THRESHOLD,

        "uncertain_minimum":
            UNCERTAIN_THRESHOLD,

        "model":
            MODEL_NAME,

        "embedding_dimensions":
            EXPECTED_EMBEDDING_DIMENSIONS,
    }

    return (
        summary,
        matches,
        uncertain,
        all_results,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # ========================================================
    # ARGUMENTS
    # ========================================================

    if len(sys.argv) != 3:

        print(
            'Usage: python search_index.py '
            '"<reference_image>" '
            '"<face_index.json>"',
            file=sys.stderr,
            flush=True,
        )

        sys.exit(1)

    reference_path = Path(
        sys.argv[1]
    ).expanduser().resolve()

    index_path = Path(
        sys.argv[2]
    ).expanduser().resolve()

    # ========================================================
    # VALIDATE REFERENCE
    # ========================================================

    if not reference_path.exists():

        raise ValueError(
            f"Reference image does not exist:\n"
            f"{reference_path}"
        )

    if not reference_path.is_file():

        raise ValueError(
            f"Reference image is not a file:\n"
            f"{reference_path}"
        )

    # ========================================================
    # VALIDATE INDEX
    # ========================================================

    if not index_path.exists():

        raise ValueError(
            f"Face index does not exist:\n"
            f"{index_path}"
        )

    if not index_path.is_file():

        raise ValueError(
            f"Face index is not a file:\n"
            f"{index_path}"
        )

    print(
        "============================================================",
        flush=True,
    )

    print(
        "FACE INDEX SEARCH",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    print(
        f"Reference: {reference_path}",
        flush=True,
    )

    print(
        f"Index:     {index_path}",
        flush=True,
    )

    print(
        f"Model:     {MODEL_NAME}",
        flush=True,
    )

    print(
        f"Threshold: {MATCH_THRESHOLD}",
        flush=True,
    )

    print(
        f"Uncertain: {UNCERTAIN_THRESHOLD}",
        flush=True,
    )

    print(
        "Supported index versions: 1, 2",
        flush=True,
    )

    # ========================================================
    # LOAD INDEX
    # ========================================================

    index_data = load_face_index(
        index_path
    )

    indexed_images = index_data.get(
        "images",
        [],
    )

    if len(
        indexed_images
    ) == 0:

        raise ValueError(
            "face_index.json contains no indexed images."
        )

    # ========================================================
    # LOAD MODEL
    # ========================================================

    app = load_model()

    # ========================================================
    # CREATE REFERENCE EMBEDDING
    # ========================================================

    (
        reference_embedding,
        reference_face_count,
    ) = create_reference_embedding(
        app,
        reference_path,
    )

    # ========================================================
    # SEARCH
    # ========================================================

    send_progress(
        "Comparing reference face with indexed faces...",
        20,
        current=0,
        total=len(indexed_images),
        progress_type="search-start",
    )

    (
        summary,
        matches,
        uncertain,
        all_results,
    ) = search_index(
        reference_embedding,
        index_data,
        index_path,
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    summary[
        "reference_faces_detected"
    ] = reference_face_count

    summary[
        "index_version"
    ] = index_data.get(
        "version",
        1,
    )

    # ========================================================
    # RESULT
    # ========================================================

    result = {

        "success": True,

        "reference": str(
            reference_path
        ),

        "index": str(
            index_path
        ),

        "model":
            MODEL_NAME,

        "embedding_dimensions":
            EXPECTED_EMBEDDING_DIMENSIONS,

        "threshold":
            MATCH_THRESHOLD,

        "uncertain_threshold":
            UNCERTAIN_THRESHOLD,

        "summary":
            summary,

        # Actual matches
        "matches":
            matches,

        # Uncertain
        "uncertain":
            uncertain,

        # All usable results
        "results":
            all_results,

        # Compatibility
        "matching_photos":
            matches,

        "matchingPhotos":
            matches,

        "uncertain_matches":
            uncertain,

        "images":
            matches,

        "photos":
            matches,
    }

    # ========================================================
    # CONSOLE OUTPUT
    # ========================================================

    print(
        "",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    print(
        "SEARCH COMPLETE",
        flush=True,
    )

    print(
        "============================================================",
        flush=True,
    )

    print(
        f"Index version:      {summary['index_version']}",
        flush=True,
    )

    print(
        f"Images found:       {summary['images_found']}",
        flush=True,
    )

    print(
        f"Images processed:   {summary['images_processed']}",
        flush=True,
    )

    print(
        f"Images with faces:  {summary['images_with_faces']}",
        flush=True,
    )

    print(
        f"Matches:            {summary['matches']}",
        flush=True,
    )

    print(
        f"Uncertain:          {summary['uncertain']}",
        flush=True,
    )

    print(
        f"No faces:           {summary['no_faces']}",
        flush=True,
    )

    print(
        f"Errors:             {summary['errors']}",
        flush=True,
    )

    print(
        f"Match threshold:    {MATCH_THRESHOLD}",
        flush=True,
    )

    print(
        f"Uncertain minimum:  {UNCERTAIN_THRESHOLD}",
        flush=True,
    )

    # ========================================================
    # MATCHING IMAGES
    # ========================================================

    print(
        "",
        flush=True,
    )

    print(
        "MATCHING IMAGES",
        flush=True,
    )

    print(
        "------------------------------------------------------------",
        flush=True,
    )

    for item in matches:

        print(
            f"{item['similarity']:.4f}  "
            f"{item['image']}",
            flush=True,
        )

    # ========================================================
    # UNCERTAIN IMAGES
    # ========================================================

    if uncertain:

        print(
            "",
            flush=True,
        )

        print(
            "UNCERTAIN IMAGES",
            flush=True,
        )

        print(
            "------------------------------------------------------------",
            flush=True,
        )

        for item in uncertain:

            print(
                f"{item['similarity']:.4f}  "
                f"{item['image']}",
                flush=True,
            )

    # ========================================================
    # COMPLETE PROGRESS
    # ========================================================

    send_progress(
        "Scan complete.",
        100,
        current=summary["images_processed"],
        total=summary["images_found"],
        progress_type="complete",
    )

    # ========================================================
    # RESULT_JSON
    #
    # main.js expects this exact prefix.
    # ========================================================

    print(
        "RESULT_JSON:" +
        json.dumps(
            result,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        ),
        flush=True,
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "ERROR: Search cancelled.",
            file=sys.stderr,
            flush=True,
        )

        sys.exit(1)

    except Exception as error:

        # ----------------------------------------------------
        # Progress error
        # ----------------------------------------------------

        try:

            send_progress(
                str(error),
                progress_type="error",
            )

        except Exception:
            pass

        # ----------------------------------------------------
        # Stderr error
        # ----------------------------------------------------

        print(
            f"ERROR:{error}",
            file=sys.stderr,
            flush=True,
        )

        sys.exit(1)