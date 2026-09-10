import os
import sys
import json
import time
from pathlib import Path

import cv2
import numpy as np

from insightface.app import FaceAnalysis


# ============================================================
# AI PERSON PHOTO FINDER
# FAST + INCREMENTAL FACE INDEX BUILDER
# ============================================================

INDEX_FILENAME = "face_index.json"

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff",
}

INDEX_VERSION = 2
MODEL_NAME = "buffalo_l"

# ------------------------------------------------------------
# Detection size
#
# 640 = best detection / slower
# 512 = recommended balance
# 384 = faster / may miss smaller faces
# ------------------------------------------------------------

DET_SIZE = (512, 512)

# Ignore extremely tiny faces.
MIN_FACE_SIZE = 40


# ============================================================
# PROGRESS
# ============================================================

def send_progress(
    message,
    percent=None,
    current=None,
    total=None,
    progress_type="indexing",
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
# IMAGE COLLECTION
# ============================================================

def collect_images(folder):
    """
    Recursively collect supported images.

    os.scandir is used instead of Path.rglob because it is
    generally faster for very large photo collections.
    """

    result = []

    def scan(directory):
        try:
            with os.scandir(directory) as entries:

                for entry in entries:

                    try:

                        if entry.is_dir(
                            follow_symlinks=False
                        ):
                            scan(entry.path)
                            continue

                        if not entry.is_file(
                            follow_symlinks=False
                        ):
                            continue

                        extension = os.path.splitext(
                            entry.name
                        )[1].lower()

                        if extension in IMAGE_EXTENSIONS:
                            result.append(
                                Path(
                                    entry.path
                                ).resolve()
                            )

                    except OSError:
                        continue

        except OSError as error:

            print(
                f"Directory scan error: "
                f"{directory}: {error}",
                file=sys.stderr,
                flush=True,
            )

    scan(str(folder))

    result.sort(
        key=lambda p: str(p).lower()
    )

    return result


# ============================================================
# FILE SIGNATURE
# ============================================================

def file_signature(path):

    stat = path.stat()

    return {
        "path": str(
            path.resolve()
        ),
        "size": int(
            stat.st_size
        ),
        "mtime_ns": int(
            stat.st_mtime_ns
        ),
    }


# ============================================================
# LOAD EXISTING INDEX
# ============================================================

def load_existing_index(index_path):

    try:

        if not index_path.exists():
            return None

        with index_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(file)

        if not isinstance(
            data,
            dict,
        ):
            return None

        if not isinstance(
            data.get("images"),
            list,
        ):
            return None

        return data

    except Exception as error:

        print(
            f"Could not load existing index: "
            f"{error}",
            file=sys.stderr,
            flush=True,
        )

        return None


# ============================================================
# INDEX COMPATIBILITY
# ============================================================

def index_metadata_is_compatible(index_data):

    if not isinstance(
        index_data,
        dict,
    ):
        return False

    if index_data.get(
        "version"
    ) != INDEX_VERSION:
        return False

    if index_data.get(
        "model"
    ) != MODEL_NAME:
        return False

    return True


# ============================================================
# NORMALIZE EXISTING RECORD
# ============================================================

def normalize_record(record):

    if not isinstance(
        record,
        dict,
    ):
        return None

    try:

        faces = record.get(
            "faces",
            [],
        )

        if not isinstance(
            faces,
            list,
        ):
            faces = []

        return {
            "path": str(
                Path(
                    record["path"]
                ).resolve()
            ),

            "size": int(
                record["size"]
            ),

            "mtime_ns": int(
                record["mtime_ns"]
            ),

            "faces": faces,
        }

    except Exception:

        return None


# ============================================================
# EXISTING RECORD MAP
# ============================================================

def build_existing_record_map(index_data):

    records_by_path = {}

    if not index_data:
        return records_by_path

    for raw_record in index_data.get(
        "images",
        [],
    ):

        record = normalize_record(
            raw_record
        )

        if record is None:
            continue

        records_by_path[
            record["path"]
        ] = record

    return records_by_path


# ============================================================
# RECORD CURRENT CHECK
# ============================================================

def record_is_current(
    record,
    image_path,
):

    try:

        signature = file_signature(
            image_path
        )

        return (
            record.get("path")
            == signature["path"]
            and
            int(
                record.get(
                    "size",
                    -1,
                )
            )
            == signature["size"]
            and
            int(
                record.get(
                    "mtime_ns",
                    -1,
                )
            )
            == signature["mtime_ns"]
        )

    except Exception:

        return False


# ============================================================
# GPU / CPU PROVIDER DETECTION
# ============================================================

def get_execution_providers():

    try:

        import onnxruntime as ort

        available = (
            ort.get_available_providers()
        )

        print(
            "ONNX Runtime providers:",
            available,
            flush=True,
        )

        # ----------------------------------------------------
        # NVIDIA CUDA
        # ----------------------------------------------------

        if (
            "CUDAExecutionProvider"
            in available
        ):

            print(
                "GPU acceleration detected: "
                "CUDAExecutionProvider",
                flush=True,
            )

            return [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]

        # ----------------------------------------------------
        # DirectML
        #
        # Useful on Windows for supported GPUs.
        # ----------------------------------------------------

        if (
            "DmlExecutionProvider"
            in available
        ):

            print(
                "GPU acceleration detected: "
                "DmlExecutionProvider",
                flush=True,
            )

            return [
                "DmlExecutionProvider",
                "CPUExecutionProvider",
            ]

        print(
            "No GPU provider available. "
            "Using CPU.",
            flush=True,
        )

    except Exception as error:

        print(
            "Could not detect GPU provider:",
            error,
            flush=True,
        )

    return [
        "CPUExecutionProvider"
    ]


# ============================================================
# LOAD INSIGHTFACE
# ============================================================

def load_model():

    send_progress(
        "Loading AI face model for indexing...",
        5,
    )

    # Force DirectML (Windows GPU) execution with CPU fallback
    providers = ['DmlExecutionProvider', 'CPUExecutionProvider']

    print(
        "Loading InsightFace with providers:",
        providers,
        flush=True,
    )

    app = FaceAnalysis(
        name=MODEL_NAME,
        providers=providers,
    )

    app.prepare(
        ctx_id=0,
        det_size=DET_SIZE,
    )

    print(
        "InsightFace model ready.",
        flush=True,
    )

    return app, providers

# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    app,
    image_path,
):
    """
    Detect faces and create normalized embeddings
    for one image.
    """

    record = {
        **file_signature(
            image_path
        ),
        "faces": [],
    }

    try:

        image = cv2.imread(
            str(image_path)
        )

        if image is None:

            print(
                f"Could not read image: "
                f"{image_path}",
                file=sys.stderr,
                flush=True,
            )

            return record, False

        faces = app.get(
            image
        )

        for face in faces:

            # ------------------------------------------------
            # Face-size filtering
            # ------------------------------------------------

            try:

                bbox = np.asarray(
                    face.bbox,
                    dtype=np.float32,
                )

                if bbox.size >= 4:

                    width = float(
                        bbox[2] - bbox[0]
                    )

                    height = float(
                        bbox[3] - bbox[1]
                    )

                    if (
                        width < MIN_FACE_SIZE
                        or
                        height < MIN_FACE_SIZE
                    ):
                        continue

            except Exception:

                pass

            # ------------------------------------------------
            # Get embedding
            # ------------------------------------------------

            embedding = np.asarray(
                face.embedding,
                dtype=np.float32,
            )

            if embedding.size == 0:
                continue

            norm = float(
                np.linalg.norm(
                    embedding
                )
            )

            if norm <= 0:
                continue

            embedding = (
                embedding / norm
            )

            record[
                "faces"
            ].append(
                embedding.astype(
                    float
                ).tolist()
            )

        return record, True

    except Exception as error:

        print(
            f"Indexing error: "
            f"{image_path}: {error}",
            file=sys.stderr,
            flush=True,
        )

        return record, False


# ============================================================
# CREATE INDEX OUTPUT
# ============================================================

def create_index_output(
    folder,
    records,
    failures,
    providers,
    created_at=None,
):

    embedding_dimensions = 0

    for record in records:

        for embedding in record.get(
            "faces",
            [],
        ):

            if embedding:

                embedding_dimensions = len(
                    embedding
                )

                break

        if embedding_dimensions:
            break

    images_with_faces = sum(
        1
        for record in records
        if record.get("faces")
    )

    faces_count = sum(
        len(
            record.get(
                "faces",
                [],
            )
        )
        for record in records
    )

    if created_at is None:
        created_at = time.time()

    return {

        "version":
            INDEX_VERSION,

        "model":
            MODEL_NAME,

        "created_at":
            created_at,

        "source_folder":
            str(
                folder.resolve()
            ),

        "embedding_dimensions":
            embedding_dimensions,

        "providers":
            providers,

        "det_size":
            list(
                DET_SIZE
            ),

        "min_face_size":
            MIN_FACE_SIZE,

        "images":
            records,

        "stats": {

            "images":
                len(records),

            "images_with_faces":
                images_with_faces,

            "faces":
                faces_count,

            "failures":
                failures,
        },
    }


# ============================================================
# SAVE INDEX SAFELY
# ============================================================

def save_index(
    index_path,
    output,
):

    temporary_path = Path(
        str(index_path)
        + ".tmp"
    )

    with temporary_path.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            output,
            file,
            ensure_ascii=False,
            separators=(
                ",",
                ":",
            ),
        )

    os.replace(
        temporary_path,
        index_path,
    )


# ============================================================
# INCREMENTAL BUILD
# ============================================================

def build_index(
    folder,
    image_paths,
    index_path,
    existing_index,
):

    existing_records = (
        build_existing_record_map(
            existing_index
        )
    )

    current_paths = {
        str(
            image_path.resolve()
        )
        for image_path in image_paths
    }

    records_by_path = {}

    images_to_process = []

    reused_count = 0

    changed_count = 0

    new_count = 0

    # --------------------------------------------------------
    # Determine reusable records
    # --------------------------------------------------------

    for image_path in image_paths:

        absolute_path = str(
            image_path.resolve()
        )

        existing_record = (
            existing_records.get(
                absolute_path
            )
        )

        if (
            existing_record is not None
            and
            record_is_current(
                existing_record,
                image_path,
            )
        ):

            records_by_path[
                absolute_path
            ] = existing_record

            reused_count += 1

        else:

            images_to_process.append(
                image_path
            )

            if existing_record is None:
                new_count += 1
            else:
                changed_count += 1

    # --------------------------------------------------------
    # Deleted images
    # --------------------------------------------------------

    deleted_count = sum(
        1
        for path in existing_records
        if path not in current_paths
    )

    total = len(
        image_paths
    )

    process_count = len(
        images_to_process
    )

    print(
        "",
        flush=True,
    )

    print(
        "=" * 60,
        flush=True,
    )

    print(
        "INCREMENTAL FACE INDEX",
        flush=True,
    )

    print(
        "=" * 60,
        flush=True,
    )

    print(
        f"Total images:       {total}",
        flush=True,
    )

    print(
        f"Already indexed:    {reused_count}",
        flush=True,
    )

    print(
        f"New images:         {new_count}",
        flush=True,
    )

    print(
        f"Changed images:     {changed_count}",
        flush=True,
    )

    print(
        f"Removed images:     {deleted_count}",
        flush=True,
    )

    print(
        "=" * 60,
        flush=True,
    )

    # --------------------------------------------------------
    # FAST PATH
    # --------------------------------------------------------

    if process_count == 0:

        records = [
            records_by_path[
                str(
                    image_path.resolve()
                )
            ]
            for image_path in image_paths
        ]

        records.sort(
            key=lambda record:
                record["path"].lower()
        )

        providers = (
            existing_index.get(
                "providers",
                [
                    "CPUExecutionProvider"
                ],
            )
            if existing_index
            else [
                "CPUExecutionProvider"
            ]
        )

        failures = sum(
            1
            for record in records
            if not record.get(
                "faces"
            )
        )

        # Preserve original creation time.
        created_at = (
            existing_index.get(
                "created_at"
            )
            if existing_index
            else None
        )

        output = create_index_output(
            folder,
            records,
            failures,
            providers,
            created_at,
        )

        save_index(
            index_path,
            output,
        )

        send_progress(
            f"Using existing face index "
            f"({total} photos).",
            95,
            current=total,
            total=total,
            progress_type="index-reused",
        )

        print(
            "FACE_INDEX_REUSED:" +
            json.dumps(
                {
                    "index":
                        str(index_path),

                    "stats":
                        output["stats"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        return output, True

    # --------------------------------------------------------
    # Load AI model ONLY when required
    # --------------------------------------------------------

    app, providers = load_model()

    processed = 0
    failures = 0

    # --------------------------------------------------------
    # Process new / changed images
    # --------------------------------------------------------

    for image_path in images_to_process:

        processed += 1

        progress = (
            10
            +
            (
                processed
                /
                max(
                    process_count,
                    1,
                )
            )
            * 80
        )

        send_progress(
            f"Indexing: "
            f"{image_path.name}",
            progress,
            current=processed,
            total=process_count,
        )

        record, success = process_image(
            app,
            image_path,
        )

        if not success:
            failures += 1

        records_by_path[
            str(
                image_path.resolve()
            )
        ] = record

    # --------------------------------------------------------
    # Rebuild final record list in collection order
    # --------------------------------------------------------

    records = []

    for image_path in image_paths:

        path_key = str(
            image_path.resolve()
        )

        record = records_by_path.get(
            path_key
        )

        if record is not None:
            records.append(
                record
            )

    records.sort(
        key=lambda record:
            str(
                record.get(
                    "path",
                    "",
                )
            ).lower()
    )

    # --------------------------------------------------------
    # Create output
    # --------------------------------------------------------

    send_progress(
        "Creating face index...",
        92,
        current=processed,
        total=process_count,
    )

    output = create_index_output(
        folder,
        records,
        failures,
        providers,
        time.time(),
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    send_progress(
        "Saving face index...",
        95,
        current=processed,
        total=process_count,
    )

    save_index(
        index_path,
        output,
    )

    return output, False


# ============================================================
# MAIN
# ============================================================

def main():

    if len(sys.argv) != 2:

        print(
            'Usage: python build_face_index.py '
            '"<photo_folder>"',
            flush=True,
        )

        sys.exit(1)

    folder = Path(
        sys.argv[1]
    ).expanduser().resolve()

    if (
        not folder.exists()
        or
        not folder.is_dir()
    ):

        raise ValueError(
            f"Photo folder does not exist: "
            f"{folder}"
        )

    # --------------------------------------------------------
    # Scan collection
    # --------------------------------------------------------

    send_progress(
        "Scanning photo collection...",
        1,
    )

    image_paths = collect_images(
        folder
    )

    if not image_paths:

        raise ValueError(
            "No supported images found "
            "in the selected folder."
        )

    total = len(
        image_paths
    )

    index_path = (
        folder
        /
        INDEX_FILENAME
    )

    print(
        f"Images discovered: {total}",
        flush=True,
    )

    # --------------------------------------------------------
    # Load existing index
    # --------------------------------------------------------

    existing = load_existing_index(
        index_path
    )

    # --------------------------------------------------------
    # FAST PATH
    #
    # If everything is unchanged, return immediately.
    # NO InsightFace model is loaded.
    # --------------------------------------------------------

    if (
        existing is not None
        and
        index_metadata_is_compatible(
            existing
        )
    ):

        existing_records = (
            build_existing_record_map(
                existing
            )
        )

        if len(existing_records) == total:

            all_current = True

            for image_path in image_paths:

                path_key = str(
                    image_path.resolve()
                )

                record = (
                    existing_records.get(
                        path_key
                    )
                )

                if (
                    record is None
                    or
                    not record_is_current(
                        record,
                        image_path,
                    )
                ):

                    all_current = False
                    break

            if all_current:

                stats = existing.get(
                    "stats",
                    {},
                )

                send_progress(
                    f"Using existing face index "
                    f"({total} photos).",
                    95,
                    current=total,
                    total=total,
                    progress_type="index-reused",
                )

                print(
                    "FACE_INDEX_REUSED:" +
                    json.dumps(
                        {
                            "index":
                                str(index_path),

                            "stats":
                                stats,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )

                send_progress(
                    "Face index ready.",
                    100,
                    current=total,
                    total=total,
                    progress_type="index-complete",
                )

                return

    # --------------------------------------------------------
    # Incremental build
    # --------------------------------------------------------

    send_progress(
        f"Checking {total} photos for changes...",
        8,
        current=0,
        total=total,
    )

    output, reused = build_index(
        folder,
        image_paths,
        index_path,
        existing,
    )

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    if reused:

        print(
            "FACE_INDEX_REUSED:" +
            json.dumps(
                {
                    "index":
                        str(index_path),

                    "stats":
                        output["stats"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    else:

        print(
            "FACE_INDEX_BUILT:" +
            json.dumps(
                {
                    "index":
                        str(index_path),

                    "stats":
                        output["stats"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

        send_progress(
            "Face index ready.",
            100,
            current=total,
            total=total,
            progress_type="index-complete",
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except Exception as error:

        send_progress(
            str(error),
            progress_type="error",
        )

        print(
            f"ERROR:{error}",
            file=sys.stderr,
            flush=True,
        )

        sys.exit(1)