// ============================================================
// AI PERSON PHOTO FINDER
// Electron Main Process
//
// Workflow:
//
// Reference Photo
//       |
//       v
// Photo Folder / ZIP
//       |
//       v
// Resolve source
//       |
//       v
// Persistent ZIP cache
//       |
//       v
// Build / Reuse incremental face_index.json
//       |
//       v
// search_index.py
//       |
//       v
// Results
//
// IMPORTANT:
// Existing renderer/preload IPC API is preserved.
// ============================================================

"use strict";

const {
    app,
    BrowserWindow,
    ipcMain,
    dialog
} = require("electron");

const path = require("path");
const fs = require("fs");
const crypto = require("crypto");
const { spawn } = require("child_process");
const AdmZip = require("adm-zip");

// ============================================================
// GPU SCANNER ENGINE ACCELERATOR
// ============================================================
const FastPhotoScanner = require("./src/batchProcessor");
const modelPath = path.join(__dirname, "models", "face_recognition.onnx");
const gpuScanner = new FastPhotoScanner(modelPath);

// ============================================================
// CONFIGURATION
// ============================================================

const INDEX_FILENAME = "face_index.json";
const RESULTS_FILENAME = "search_results.json";

const SUPPORTED_IMAGE_EXTENSIONS = new Set([
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".tif",
    ".tiff"
]);

// ============================================================
// GLOBAL STATE
// ============================================================

let mainWindow = null;
let lastScanData = null;
let currentPythonProcess = null;

// Source cache is only used to avoid repeated directory walks
// during the same app session.
const sourceCache = new Map();

// ============================================================
// BASIC HELPERS
// ============================================================

function safeExists(filePath) {
    try {
        return Boolean(filePath) && fs.existsSync(filePath);
    } catch {
        return false;
    }
}

function isDirectory(filePath) {
    try {
        return safeExists(filePath) &&
            fs.statSync(filePath).isDirectory();
    } catch {
        return false;
    }
}

function isFile(filePath) {
    try {
        return safeExists(filePath) &&
            fs.statSync(filePath).isFile();
    } catch {
        return false;
    }
}

function normalizePath(filePath) {
    return path.resolve(String(filePath));
}

function isSupportedImage(filePath) {
    return SUPPORTED_IMAGE_EXTENSIONS.has(
        path.extname(filePath).toLowerCase()
    );
}

// ============================================================
// FILE SIGNATURE
// ============================================================

function getFileSignature(filePath) {
    const stats = fs.statSync(filePath);

    return {
        size: Number(stats.size),
        mtimeMs: Number(stats.mtimeMs)
    };
}

// ============================================================
// DIRECTORY IMAGE SCAN
// ============================================================

function getImageFilesFromDirectory(
    folderPath,
    onProgress = null
) {
    const imageFiles = [];

    if (!isDirectory(folderPath)) {
        return imageFiles;
    }

    const pending = [folderPath];

    while (pending.length > 0) {
        const currentDirectory = pending.pop();

        let entries;

        try {
            entries = fs.readdirSync(
                currentDirectory,
                {
                    withFileTypes: true
                }
            );
        } catch (error) {
            console.warn(
                "Unable to read directory:",
                currentDirectory,
                error.message
            );

            continue;
        }

        for (const entry of entries) {
            const fullPath = path.join(
                currentDirectory,
                entry.name
            );

            if (entry.isDirectory()) {
                pending.push(fullPath);
                continue;
            }

            if (!entry.isFile()) {
                continue;
            }

            if (isSupportedImage(fullPath)) {
                imageFiles.push(fullPath);
            }
        }

        if (typeof onProgress === "function") {
            onProgress(imageFiles.length);
        }
    }

    return imageFiles;
}

// ============================================================
// DIRECTORY CACHE
// ============================================================

function getDirectoryCacheKey(folderPath) {
    const absolutePath = normalizePath(folderPath);

    try {
        const stats = fs.statSync(absolutePath);

        return [
            absolutePath,
            Number(stats.mtimeMs),
            Number(stats.size)
        ].join("|");
    } catch {
        return absolutePath;
    }
}

function getCachedImageFiles(folderPath) {
    const key = getDirectoryCacheKey(folderPath);

    const cached = sourceCache.get(key);

    if (
        cached &&
        Array.isArray(cached.imageFiles)
    ) {
        return cached.imageFiles;
    }

    const imageFiles =
        getImageFilesFromDirectory(folderPath);

    sourceCache.set(
        key,
        {
            folderPath: normalizePath(folderPath),
            imageFiles,
            imageCount: imageFiles.length,
            createdAt: Date.now()
        }
    );

    return imageFiles;
}

function countImagesInDirectory(folderPath) {
    return getCachedImageFiles(folderPath).length;
}

// ============================================================
// INVALIDATE SOURCE CACHE
// ============================================================

function invalidateSourceCache(folderPath = null) {
    if (!folderPath) {
        sourceCache.clear();
        return;
    }

    const absolutePath =
        normalizePath(folderPath);

    for (const [key, value] of sourceCache.entries()) {
        if (
            value &&
            value.folderPath &&
            normalizePath(value.folderPath) === absolutePath
        ) {
            sourceCache.delete(key);
        }
    }
}

// ============================================================
// ZIP CACHE
// ============================================================

function getZipCacheRoot() {
    const root = path.join(
        app.getPath("userData"),
        "zip-cache"
    );

    fs.mkdirSync(
        root,
        {
            recursive: true
        }
    );

    return root;
}

function createZipCacheKey(zipPath) {
    const absolutePath =
        normalizePath(zipPath);

    const signature =
        getFileSignature(absolutePath);

    const input =
        absolutePath +
        "|" +
        signature.size +
        "|" +
        signature.mtimeMs;

    return crypto
        .createHash("sha256")
        .update(input)
        .digest("hex")
        .slice(0, 32);
}

function getZipCacheFolder(zipPath) {
    return path.join(
        getZipCacheRoot(),
        createZipCacheKey(zipPath)
    );
}

// ============================================================
// SAFE DIRECTORY DELETE
// ============================================================

function removeDirectorySafe(directoryPath) {
    try {
        if (
            !directoryPath ||
            !fs.existsSync(directoryPath)
        ) {
            return;
        }

        fs.rmSync(
            directoryPath,
            {
                recursive: true,
                force: true
            }
        );
    } catch (error) {
        console.warn(
            "Could not remove directory:",
            directoryPath,
            error.message
        );
    }
}

// ============================================================
// ZIP PATH SAFETY
// ============================================================

function validateZipEntries(
    extractionPath,
    entries
) {
    const extractionRoot =
        path.resolve(extractionPath);

    for (const entry of entries) {
        const entryName = String(
            entry.entryName || ""
        ).replace(/\\/g, "/");

        if (
            !entryName ||
            entryName.endsWith("/")
        ) {
            continue;
        }

        const targetPath =
            path.resolve(
                extractionPath,
                entryName
            );

        const relativePath =
            path.relative(
                extractionRoot,
                targetPath
            );

        if (
            relativePath.startsWith("..") ||
            path.isAbsolute(relativePath)
        ) {
            throw new Error(
                "The ZIP contains an unsafe file path and cannot be extracted."
            );
        }
    }
}

// ============================================================
// ZIP EXTRACTION WITH PERSISTENT CACHE
// ============================================================

function extractZipCached(zipPath) {
    if (!isFile(zipPath)) {
        throw new Error(
            "ZIP file does not exist."
        );
    }

    if (
        path.extname(zipPath).toLowerCase() !== ".zip"
    ) {
        throw new Error(
            "Selected file is not a ZIP file."
        );
    }

    const absoluteZipPath =
        normalizePath(zipPath);

    const cacheFolder =
        getZipCacheFolder(
            absoluteZipPath
        );

    const markerFile =
        path.join(
            cacheFolder,
            ".zip-cache-complete"
        );

    // ========================================================
    // CHECK EXISTING CACHE
    // ========================================================

    if (fs.existsSync(markerFile)) {
        try {
            const marker =
                JSON.parse(
                    fs.readFileSync(
                        markerFile,
                        "utf8"
                    )
                );

            const currentSignature =
                getFileSignature(
                    absoluteZipPath
                );

            if (
                Number(marker.sourceSize) ===
                    currentSignature.size &&
                Number(marker.sourceMtimeMs) ===
                    currentSignature.mtimeMs
            ) {
                const cachedImages =
                    getCachedImageFiles(
                        cacheFolder
                    );

                if (cachedImages.length > 0) {
                    console.log(
                        "=================================================="
                    );

                    console.log(
                        "USING CACHED ZIP EXTRACTION"
                    );

                    console.log(
                        "ZIP:",
                        absoluteZipPath
                    );

                    console.log(
                        "Cache:",
                        cacheFolder
                    );

                    console.log(
                        "Images:",
                        cachedImages.length
                    );

                    console.log(
                        "=================================================="
                    );

                    return cacheFolder;
                }
            }
        } catch (error) {
            console.warn(
                "ZIP cache validation failed:",
                error.message
            );
        }

        removeDirectorySafe(cacheFolder);
        invalidateSourceCache(cacheFolder);
    }

    // ========================================================
    // EXTRACT ZIP
    // ========================================================

    fs.mkdirSync(
        cacheFolder,
        {
            recursive: true
        }
    );

    console.log(
        "=================================================="
    );

    console.log(
        "EXTRACTING ZIP"
    );

    console.log(
        "ZIP:",
        absoluteZipPath
    );

    console.log(
        "Cache:",
        cacheFolder
    );

    console.log(
        "=================================================="
    );

    let zip;

    try {
        zip = new AdmZip(
            absoluteZipPath
        );
    } catch (error) {
        removeDirectorySafe(cacheFolder);

        throw new Error(
            "Unable to open ZIP file: " +
            error.message
        );
    }

    const entries =
        zip.getEntries();

    if (
        !entries ||
        entries.length === 0
    ) {
        removeDirectorySafe(cacheFolder);

        throw new Error(
            "The selected ZIP file is empty."
        );
    }

    validateZipEntries(
        cacheFolder,
        entries
    );

    try {
        zip.extractAllTo(
            cacheFolder,
            true
        );
    } catch (error) {
        removeDirectorySafe(cacheFolder);

        throw new Error(
            "Could not extract ZIP file: " +
            error.message
        );
    }

    invalidateSourceCache(
        cacheFolder
    );

    const imageFiles =
        getCachedImageFiles(
            cacheFolder
        );

    const imageCount =
        imageFiles.length;

    if (imageCount <= 0) {
        removeDirectorySafe(
            cacheFolder
        );

        throw new Error(
            "The selected ZIP contains no supported images."
        );
    }

    const signature =
        getFileSignature(
            absoluteZipPath
        );

    fs.writeFileSync(
        markerFile,
        JSON.stringify(
            {
                sourceZip:
                    absoluteZipPath,

                sourceSize:
                    signature.size,

                sourceMtimeMs:
                    signature.mtimeMs,

                imageCount,

                createdAt:
                    Date.now()
            },
            null,
            2
        ),
        "utf8"
    );

    console.log(
        `ZIP extraction complete. Images found: ${imageCount}`
    );

    return cacheFolder;
}

// ============================================================
// RESOLVE PHOTO SOURCE
// ============================================================

function resolvePhotoSource(
    sourcePath,
    sourceType = null
) {
    if (!sourcePath) {
        throw new Error(
            "No photo source selected."
        );
    }

    const absolutePath =
        normalizePath(sourcePath);

    if (!safeExists(absolutePath)) {
        throw new Error(
            "Selected photo source does not exist."
        );
    }

    const stats =
        fs.statSync(absolutePath);

    // ========================================================
    // FOLDER
    // ========================================================

    if (stats.isDirectory()) {
        const imageFiles =
            getCachedImageFiles(
                absolutePath
            );

        if (imageFiles.length <= 0) {
            throw new Error(
                "No supported images found in the selected folder."
            );
        }

        return {
            path: absolutePath,
            type: "folder",
            originalPath: absolutePath,
            imageFiles,
            count: imageFiles.length
        };
    }

    // ========================================================
    // ZIP
    // ========================================================

    if (
        stats.isFile() &&
        path.extname(
            absolutePath
        ).toLowerCase() === ".zip"
    ) {
        const extractedPath =
            extractZipCached(
                absolutePath
            );

        const imageFiles =
            getCachedImageFiles(
                extractedPath
            );

        if (imageFiles.length <= 0) {
            throw new Error(
                "The selected ZIP contains no supported images."
            );
        }

        return {
            path: extractedPath,
            type: "zip",
            originalPath: absolutePath,
            imageFiles,
            count: imageFiles.length
        };
    }

    throw new Error(
        "Selected photo source is not a folder or ZIP file."
    );
}

// ============================================================
// CREATE WINDOW
// ============================================================

function createWindow() {
    mainWindow =
        new BrowserWindow({
            width: 1400,
            height: 850,
            minWidth: 1000,
            minHeight: 650,
            backgroundColor: "#f5f7fb",

            webPreferences: {
                preload: path.join(
                    __dirname,
                    "preload.js"
                ),

                contextIsolation: true,
                nodeIntegration: false
            }
        });

    mainWindow.loadFile(
        path.join(
            __dirname,
            "src",
            "index.html"
        )
    );

    mainWindow.on(
        "closed",
        () => {
            mainWindow = null;
        }
    );
}

// ============================================================
// APP READY
// ============================================================

app.whenReady().then(() => {
    createWindow();

    // Initialize GPU batch engine on boot
    gpuScanner.init().catch(err => {
        console.warn("GPU Scanner Initialization Note:", err.message);
    });

    app.on(
        "activate",
        () => {
            if (
                BrowserWindow
                    .getAllWindows()
                    .length === 0
            ) {
                createWindow();
            }
        }
    );
});

// ============================================================
// APP CLOSE
// ============================================================

app.on(
    "before-quit",
    () => {
        if (currentPythonProcess) {
            try {
                currentPythonProcess.kill();
            } catch (error) {
                console.warn(
                    "Could not stop Python process:",
                    error.message
                );
            }

            currentPythonProcess = null;
        }
    }
);

app.on(
    "window-all-closed",
    () => {
        if (
            process.platform !== "darwin"
        ) {
            app.quit();
        }
    }
);
// ============================================================
// SEND PROGRESS TO RENDERER
// ============================================================

function sendScanProgress(data) {
    if (
        !mainWindow ||
        mainWindow.isDestroyed()
    ) {
        return;
    }

    mainWindow.webContents.send(
        "scan-progress",
        data
    );
}

// ============================================================
// PYTHON OUTPUT PARSER
// ============================================================

function parsePythonOutput(
    text,
    state,
    progressTransform = null
) {
    state.stdout += text;
    state.buffer += text;

    const lines =
        state.buffer.split(/\r?\n/);

    state.buffer =
        lines.pop() || "";

    for (const line of lines) {
        const trimmed =
            line.trim();

        if (!trimmed) {
            continue;
        }

        // ====================================================
        // PROGRESS_JSON
        // ====================================================

        if (
            trimmed.startsWith(
                "PROGRESS_JSON:"
            )
        ) {
            try {
                const data =
                    JSON.parse(
                        trimmed.slice(
                            "PROGRESS_JSON:".length
                        )
                    );

                state.progressEvents.push(
                    data
                );

                const progressData = {
                    ...data
                };

                if (
                    typeof progressTransform ===
                        "function" &&
                    typeof progressData.percent ===
                        "number"
                ) {
                    progressData.percent =
                        progressTransform(
                            progressData.percent
                        );
                }

                sendScanProgress(
                    progressData
                );
            } catch (error) {
                console.warn(
                    "Invalid progress JSON:",
                    trimmed
                );
            }

            continue;
        }

        // ====================================================
        // RESULT_JSON
        // ====================================================

        if (
            trimmed.startsWith(
                "RESULT_JSON:"
            )
        ) {
            try {
                state.resultData =
                    JSON.parse(
                        trimmed.slice(
                            "RESULT_JSON:".length
                        )
                    );
            } catch (error) {
                console.warn(
                    "Invalid result JSON:",
                    trimmed
                );
            }

            continue;
        }

        // ====================================================
        // FACE INDEX STATUS
        // ====================================================

        if (
            trimmed.startsWith(
                "FACE_INDEX_BUILT:"
            ) ||
            trimmed.startsWith(
                "FACE_INDEX_REUSED:"
            )
        ) {
            try {
                const colonIndex =
                    trimmed.indexOf(":");

                state.indexInfo =
                    JSON.parse(
                        trimmed.slice(
                            colonIndex + 1
                        )
                    );

                state.indexReused =
                    trimmed.startsWith(
                        "FACE_INDEX_REUSED:"
                    );
            } catch (error) {
                console.warn(
                    "Invalid face index information:",
                    trimmed
                );
            }

            continue;
        }

        // ====================================================
        // NORMAL PYTHON OUTPUT
        // ====================================================

        console.log(
            "[Python]",
            trimmed
        );
    }
}

// ============================================================
// RUN PYTHON SCRIPT
// ============================================================

function runPythonScript(
    pythonPath,
    scriptPath,
    args,
    progressTransform = null
) {
    return new Promise(
        (resolve) => {
            const state = {
                stdout: "",
                buffer: "",
                stderr: "",
                resultData: null,
                indexInfo: null,
                indexReused: false,
                progressEvents: []
            };

            let settled = false;

            function finish(value) {
                if (settled) {
                    return;
                }

                settled = true;

                if (
                    currentPythonProcess ===
                    pythonProcess
                ) {
                    currentPythonProcess = null;
                }

                resolve(value);
            }

            let pythonProcess;

            try {
                pythonProcess =
                    spawn(
                        pythonPath,
                        [
                            "-u",
                            scriptPath,
                            ...args
                        ],
                        {
                            cwd: __dirname,
                            windowsHide: true,

                            env: {
                                ...process.env,

                                PYTHONUNBUFFERED:
                                    "1"
                            }
                        }
                    );

                currentPythonProcess =
                    pythonProcess;
            } catch (error) {
                finish({
                    success: false,
                    error: error.message,
                    ...state
                });

                return;
            }

            // =================================================
            // STDOUT
            // =================================================

            pythonProcess.stdout.on(
                "data",
                (data) => {
                    parsePythonOutput(
                        data.toString(),
                        state,
                        progressTransform
                    );
                }
            );

            // =================================================
            // STDERR
            // =================================================

            pythonProcess.stderr.on(
                "data",
                (data) => {
                    const text =
                        data.toString();

                    state.stderr += text;

                    console.error(
                        "[Python Error]",
                        text.trim()
                    );
                }
            );

            // =================================================
            // PROCESS ERROR
            // =================================================

            pythonProcess.on(
                "error",
                (error) => {
                    finish({
                        success: false,
                        error: error.message,
                        ...state
                    });
                }
            );

            // =================================================
            // PROCESS CLOSE
            // =================================================

            pythonProcess.on(
                "close",
                (code) => {
                    if (
                        state.buffer.trim()
                    ) {
                        parsePythonOutput(
                            "\n",
                            state,
                            progressTransform
                        );
                    }

                    finish({
                        success:
                            code === 0,

                        code,

                        ...state
                    });
                }
            );
        }
    );
}

// ============================================================
// PYTHON EXECUTABLE
// ============================================================

function getPythonExecutable() {
    const candidates = [
        path.join(
            __dirname,
            ".venv",
            "Scripts",
            "python.exe"
        ),

        path.join(
            __dirname,
            ".venv-model",
            "Scripts",
            "python.exe"
        ),

        path.join(
            __dirname,
            "venv",
            "Scripts",
            "python.exe"
        ),

        path.join(
            __dirname,
            ".venv",
            "bin",
            "python"
        ),

        path.join(
            __dirname,
            ".venv-model",
            "bin",
            "python"
        ),

        path.join(
            __dirname,
            "venv",
            "bin",
            "python"
        )
    ];

    for (const candidate of candidates) {
        if (fs.existsSync(candidate)) {
            console.log(
                "Using Python:",
                candidate
            );

            return candidate;
        }
    }

    console.log(
        "No project virtual environment found."
    );

    console.log(
        "Falling back to system Python."
    );

    return process.platform === "win32"
        ? "python.exe"
        : "python3";
}

// ============================================================
// FACE INDEX PATH
// ============================================================

function getFaceIndexPath(sourceFolder) {
    return path.join(
        sourceFolder,
        INDEX_FILENAME
    );
}

function findFaceIndex(sourceFolder) {
    const indexPath =
        getFaceIndexPath(
            sourceFolder
        );

    return fs.existsSync(indexPath)
        ? indexPath
        : null;
}

// ============================================================
// SELECT FOLDER
// ============================================================

ipcMain.handle(
    "select-folder",
    async () => {
        try {
            const result =
                await dialog.showOpenDialog(
                    mainWindow,
                    {
                        title:
                            "Select Photo Collection",

                        properties: [
                            "openDirectory"
                        ]
                    }
                );

            if (
                result.canceled ||
                result.filePaths.length === 0
            ) {
                return null;
            }

            const selectedFolder =
                result.filePaths[0];

            console.log(
                "Selected folder:",
                selectedFolder
            );

            return selectedFolder;
        } catch (error) {
            console.error(
                "Folder selection error:",
                error
            );

            return null;
        }
    }
);

// ============================================================
// SELECT ZIP
// ============================================================

ipcMain.handle(
    "select-zip",
    async () => {
        try {
            const result =
                await dialog.showOpenDialog(
                    mainWindow,
                    {
                        title:
                            "Select Photo ZIP File",

                        properties: [
                            "openFile"
                        ],

                        filters: [
                            {
                                name:
                                    "ZIP Files",

                                extensions: [
                                    "zip"
                                ]
                            }
                        ]
                    }
                );

            if (
                result.canceled ||
                result.filePaths.length === 0
            ) {
                return null;
            }

            const selectedZip =
                result.filePaths[0];

            console.log(
                "Selected ZIP:",
                selectedZip
            );

            return selectedZip;
        } catch (error) {
            console.error(
                "ZIP selection error:",
                error
            );

            return null;
        }
    }
);

// ============================================================
// SELECT PHOTO SOURCE
// FOLDER OR ZIP
// ============================================================

ipcMain.handle(
    "select-photo-source",
    async () => {
        try {
            const result =
                await dialog.showOpenDialog(
                    mainWindow,
                    {
                        title:
                            "Select Photo Folder or ZIP File",

                        properties: [
                            "openFile",
                            "openDirectory"
                        ],

                        filters: [
                            {
                                name:
                                    "ZIP Files",

                                extensions: [
                                    "zip"
                                ]
                            },

                            {
                                name:
                                    "All Files",

                                extensions: [
                                    "*"
                                ]
                            }
                        ]
                    }
                );

            if (
                result.canceled ||
                result.filePaths.length === 0
            ) {
                return null;
            }

            const selectedPath =
                result.filePaths[0];

            const stats =
                fs.statSync(
                    selectedPath
                );

            if (stats.isDirectory()) {
                return {
                    path:
                        selectedPath,

                    type:
                        "folder"
                };
            }

            if (
                stats.isFile() &&
                path.extname(
                    selectedPath
                ).toLowerCase() === ".zip"
            ) {
                return {
                    path:
                        selectedPath,

                    type:
                        "zip"
                };
            }

            return null;
        } catch (error) {
            console.error(
                "Photo source selection error:",
                error
            );

            return null;
        }
    }
);

// ============================================================
// PREPARE PHOTO SOURCE
// ============================================================

ipcMain.handle(
    "prepare-photo-source",
    async (
        event,
        sourcePath,
        sourceType
    ) => {
        try {
            const resolved =
                resolvePhotoSource(
                    sourcePath,
                    sourceType
                );

            return {
                success: true,

                path:
                    resolved.path,

                type:
                    resolved.type,

                originalPath:
                    resolved.originalPath,

                count:
                    resolved.count
            };
        } catch (error) {
            console.error(
                "Preparing photo source failed:",
                error
            );

            return {
                success: false,

                error:
                    error.message
            };
        }
    }
);

// ============================================================
// SELECT REFERENCE PHOTO
// ============================================================

ipcMain.handle(
    "select-reference",
    async () => {
        try {
            const result =
                await dialog.showOpenDialog(
                    mainWindow,
                    {
                        title:
                            "Select Reference Photo",

                        properties: [
                            "openFile"
                        ],

                        filters: [
                            {
                                name:
                                    "Images",

                                extensions: [
                                    "jpg",
                                    "jpeg",
                                    "png",
                                    "webp",
                                    "bmp",
                                    "gif",
                                    "tif",
                                    "tiff"
                                ]
                            }
                        ]
                    }
                );

            if (
                result.canceled ||
                result.filePaths.length === 0
            ) {
                return null;
            }

            const referencePath =
                result.filePaths[0];

            console.log(
                "Selected reference:",
                referencePath
            );

            return referencePath;
        } catch (error) {
            console.error(
                "Reference selection error:",
                error
            );

            return null;
        }
    }
);

// ============================================================
// SAVE CAMERA PHOTO / SNAPSHOT
// ============================================================

ipcMain.handle(
    "save-camera-photo",
    async (event, dataUrl) => {
        try {
            if (!dataUrl) return null;

            // Remove Base64 metadata prefix
            const base64Data = dataUrl.replace(/^data:image\/\w+;base64,/, "");
            const buffer = Buffer.from(base64Data, "base64");

            // Ensure the destination 'processed' directory exists
            const outputDir = path.join(__dirname, "processed");
            if (!fs.existsSync(outputDir)) {
                fs.mkdirSync(outputDir, { recursive: true });
            }

            // Save snapshot as a timestamped JPEG image
            const filePath = path.join(outputDir, `camera_ref_${Date.now()}.jpg`);
            fs.writeFileSync(filePath, buffer);

            console.log("Saved camera snapshot:", filePath);

            return filePath;
        } catch (error) {
            console.error("Camera photo saving error:", error);
            return null;
        }
    }
);

// ============================================================
// COUNT IMAGES
// ============================================================

ipcMain.handle(
    "count-images",
    async (
        event,
        folderPath
    ) => {
        try {
            if (!folderPath) {
                return 0;
            }

            const resolved =
                resolvePhotoSource(
                    folderPath
                );

            return resolved.count;
        } catch (error) {
            console.error(
                "Image counting failed:",
                error
            );

            return 0;
        }
    }
);

// ============================================================
// GET IMAGE FILES
// ============================================================

ipcMain.handle(
    "get-image-files",
    async (
        event,
        folderPath
    ) => {
        try {
            if (!folderPath) {
                return [];
            }

            const resolved =
                resolvePhotoSource(
                    folderPath
                );

            return resolved.imageFiles;
        } catch (error) {
            console.error(
                "Getting image files failed:",
                error
            );

            return [];
        }
    }
);

// ============================================================
// COUNT PHOTO SOURCE
// ============================================================

ipcMain.handle(
    "count-photo-source",
    async (
        event,
        sourcePath
    ) => {
        try {
            if (!sourcePath) {
                return 0;
            }

            const resolved =
                resolvePhotoSource(
                    sourcePath
                );

            return resolved.count;
        } catch (error) {
            console.error(
                "Photo source counting failed:",
                error
            );

            return 0;
        }
    }
);

// ============================================================
// GET PHOTO SOURCE IMAGE FILES
// ============================================================

ipcMain.handle(
    "get-photo-source-images",
    async (
        event,
        sourcePath
    ) => {
        try {
            if (!sourcePath) {
                return [];
            }

            const resolved =
                resolvePhotoSource(
                    sourcePath
                );

            return resolved.imageFiles;
        } catch (error) {
            console.error(
                "Getting photo source images failed:",
                error
            );

            return [];
        }
    }
);

// ============================================================
// RUN FACE SCAN
// ============================================================

ipcMain.handle(
    "run-face-scan",
    async (
        event,
        referencePath,
        sourcePath
    ) => {
        try {
            // =================================================
            // PREVENT MULTIPLE PYTHON SCANS
            // =================================================

            if (currentPythonProcess) {
                return {
                    success: false,

                    error:
                        "A face scan is already running. Please wait for it to finish."
                };
            }

            // =================================================
            // VALIDATION
            // =================================================

            if (!referencePath) {
                return {
                    success: false,

                    error:
                        "No reference image selected."
                };
            }

            if (!sourcePath) {
                return {
                    success: false,

                    error:
                        "No photo collection selected."
                };
            }

            if (!isFile(referencePath)) {
                return {
                    success: false,

                    error:
                        "Reference image does not exist."
                };
            }

            if (!isSupportedImage(referencePath)) {
                return {
                    success: false,

                    error:
                        "Reference file is not a supported image."
                };
            }

            // =================================================
            // PYTHON
            // =================================================

            const pythonPath =
                getPythonExecutable();

            const buildScriptPath =
                path.join(
                    __dirname,
                    "build_face_index.py"
                );

            const searchScriptPath =
                path.join(
                    __dirname,
                    "search_index.py"
                );

            // =================================================
            // CHECK PYTHON
            // =================================================

            if (
                path.isAbsolute(pythonPath) &&
                !fs.existsSync(pythonPath)
            ) {
                return {
                    success: false,

                    error:
                        "Python executable not found:\n" +
                        pythonPath
                };
            }

            // =================================================
            // CHECK BUILD SCRIPT
            // =================================================

            if (
                !fs.existsSync(
                    buildScriptPath
                )
            ) {
                return {
                    success: false,

                    error:
                        "build_face_index.py not found:\n" +
                        buildScriptPath
                };
            }

            // =================================================
            // CHECK SEARCH SCRIPT
            // =================================================

            if (
                !fs.existsSync(
                    searchScriptPath
                )
            ) {
                return {
                    success: false,

                    error:
                        "search_index.py not found:\n" +
                        searchScriptPath
                };
            }

            // =================================================
            // RESOLVE SOURCE ONCE
            // =================================================

            sendScanProgress({
                type:
                    "preparing",

                message:
                    "Preparing photo collection...",

                percent:
                    1
            });

            const resolved =
                resolvePhotoSource(
                    sourcePath
                );

            const searchFolder =
                resolved.path;

            const imageCount =
                resolved.count;

            const extractedPath =
                resolved.type === "zip"
                    ? resolved.path
                    : null;

            // =================================================
            // LOG
            // =================================================

            console.log("");

            console.log(
                "=================================================="
            );

            console.log(
                "AI PERSON PHOTO FINDER SEARCH"
            );

            console.log(
                "=================================================="
            );

            console.log(
                "Reference:",
                referencePath
            );

            console.log(
                "Original source:",
                sourcePath
            );

            console.log(
                "Search folder:",
                searchFolder
            );

            console.log(
                "Images:",
                imageCount
            );

            console.log(
                "Python:",
                pythonPath
            );

            // =================================================
            // STEP 1
            // INCREMENTAL FACE INDEX
            // =================================================

            sendScanProgress({
                type:
                    "indexing",

                message:
                    "Checking face index...",

                percent:
                    2,

                current:
                    0,

                total:
                    imageCount
            });

            const buildResult =
                await runPythonScript(
                    pythonPath,
                    buildScriptPath,
                    [
                        searchFolder
                    ],
                    (percent) => {
                        const numeric =
                            Number(percent);

                        if (
                            !Number.isFinite(
                                numeric
                            )
                        ) {
                            return 2;
                        }

                        return Math.max(
                            2,
                            Math.min(
                                45,
                                numeric * 0.45
                            )
                        );
                    }
                );

            // =================================================
            // BUILD FAILED
            // =================================================

            if (
                !buildResult.success
            ) {
                const errorMessage =
                    buildResult.stderr ||
                    buildResult.error ||
                    `Face index builder exited with code ${buildResult.code}.`;

                sendScanProgress({
                    type:
                        "error",

                    message:
                        errorMessage
                });

                return {
                    success:
                        false,

                    error:
                        errorMessage,

                    stdout:
                        buildResult.stdout,

                    stderr:
                        buildResult.stderr
                };
            }

            // =================================================
            // INDEX PATH
            // =================================================

            const indexPath =
                buildResult.indexInfo?.index ||
                getFaceIndexPath(
                    searchFolder
                );

            // =================================================
            // IMPORTANT:
            // Normalize the index path to an absolute path.
            // =================================================

            const absoluteIndexPath =
                normalizePath(indexPath);

            console.log(
                "Face index path:",
                absoluteIndexPath
            );

            // =================================================
            // CHECK INDEX FILE
            // =================================================

            if (
                !fs.existsSync(
                    absoluteIndexPath
                )
            ) {
                return {
                    success:
                        false,

                    error:
                        "Face index builder completed but face_index.json was not created.",

                    stdout:
                        buildResult.stdout
                };
            }

            if (
                !isFile(
                    absoluteIndexPath
                )
            ) {
                return {
                    success:
                        false,

                    error:
                        "Face index path exists but is not a file:\n" +
                        absoluteIndexPath
                };
            }

            // =================================================
            // INDEX STATUS
            // =================================================

            if (
                buildResult.indexReused
            ) {
                console.log(
                    "FACE INDEX REUSED"
                );

                sendScanProgress({
                    type:
                        "index-reused",

                    message:
                        "Existing face index is up to date. Starting fast search...",

                    percent:
                        45,

                    current:
                        imageCount,

                    total:
                        imageCount
                });
            } else {
                console.log(
                    "FACE INDEX BUILT / UPDATED"
                );

                sendScanProgress({
                    type:
                        "index-complete",

                    message:
                        "Face index ready. Starting fast search...",

                    percent:
                        45,

                    current:
                        imageCount,

                    total:
                        imageCount
                });
            }

            // =================================================
            // STEP 2
            // SEARCH EXISTING INDEX
            // =================================================

            console.log("");

            console.log(
                "=================================================="
            );

            console.log(
                "FAST FACE INDEX SEARCH"
            );

            console.log(
                "=================================================="
            );

            console.log(
                "Reference image:",
                referencePath
            );

            console.log(
                "FACE INDEX FILE:",
                absoluteIndexPath
            );

            // =================================================
            // IMPORTANT FIX
            //
            // search_index.py expects:
            //
            // argument 1 = reference image
            // argument 2 = face_index.json
            //
            // DO NOT PASS searchFolder HERE.
            // =================================================

            const searchResult =
                await runPythonScript(
                    pythonPath,
                    searchScriptPath,
                    [
                        referencePath,
                        absoluteIndexPath
                    ],
                    (percent) => {
                        const numeric =
                            Number(percent);

                        if (
                            !Number.isFinite(
                                numeric
                            )
                        ) {
                            return 45;
                        }

                        return Math.max(
                            45,
                            Math.min(
                                100,
                                45 +
                                numeric * 0.55
                            )
                        );
                    }
                );

            // =================================================
            // SEARCH FAILED
            // =================================================

            if (
                !searchResult.success
            ) {
                const pythonError =
                    searchResult.resultData?.error ||
                    searchResult.stderr ||
                    searchResult.error ||
                    `Face search exited with code ${searchResult.code}.`;

                sendScanProgress({
                    type:
                        "error",

                    message:
                        pythonError
                });

                return {
                    success:
                        false,

                    error:
                        pythonError,

                    data:
                        searchResult.resultData,

                    stdout:
                        searchResult.stdout,

                    stderr:
                        searchResult.stderr
                };
            }

            // =================================================
            // RESULT REQUIRED
            // =================================================

            if (
                !searchResult.resultData
            ) {
                return {
                    success:
                        false,

                    error:
                        "Search completed but no RESULT_JSON response was received.",

                    stdout:
                        searchResult.stdout
                };
            }

            // =================================================
            // SAVE RESULT IN MEMORY
            // =================================================

            lastScanData =
                searchResult.resultData;

            // =================================================
            // ALSO SAVE RESULT JSON
            // =================================================

            try {
                const resultFilePath =
                    path.join(
                        searchFolder,
                        RESULTS_FILENAME
                    );

                fs.writeFileSync(
                    resultFilePath,
                    JSON.stringify(
                        searchResult.resultData,
                        null,
                        2
                    ),
                    "utf8"
                );
            } catch (error) {
                console.warn(
                    "Could not save search_results.json:",
                    error.message
                );
            }

            // =================================================
            // SUMMARY
            // =================================================

            const summary =
                searchResult
                    .resultData
                    .summary || {};

            const processed =
                Number(
                    summary.images_processed ||
                    imageCount
                );

            const found =
                Number(
                    summary.images_found ||
                    0
                );

            sendScanProgress({
                type:
                    "complete",

                message:
                    "Scan complete.",

                percent:
                    100,

                current:
                    processed,

                total:
                    imageCount
            });

            // =================================================
            // LOG SUMMARY
            // =================================================

            console.log("");

            console.log(
                "=================================================="
            );

            console.log(
                "SEARCH COMPLETE"
            );

            console.log(
                "=================================================="
            );

            console.log(
                "Matches:",
                summary.matches || 0
            );

            console.log(
                "Uncertain:",
                summary.uncertain || 0
            );

            console.log(
                "No matches:",
                summary.no_matches || 0
            );

            console.log(
                "Images found:",
                found
            );

            // =================================================
            // RETURN
            // =================================================

            return {
                success:
                    true,

                data:
                    searchResult.resultData,

                indexPath:
                    absoluteIndexPath,

                sourcePath:
                    searchFolder,

                originalSourcePath:
                    sourcePath,

                extractedPath,

                indexReused:
                    Boolean(
                        buildResult.indexReused
                    )
            };

        } catch (error) {
            console.error(
                "Face scan error:",
                error
            );

            sendScanProgress({
                type:
                    "error",

                message:
                    error.message
            });

            return {
                success:
                    false,

                error:
                    error.message
            };
        }
    }
);

// ============================================================
// GET SCAN RESULTS
// ============================================================

ipcMain.handle(
    "get-scan-results",
    async () => {
        try {
            if (lastScanData) {
                return {
                    success:
                        true,

                    data:
                        lastScanData
                };
            }

            return {
                success:
                    false,

                error:
                    "No scan results found."
            };
        } catch (error) {
            console.error(
                "Could not read scan results:",
                error
            );

            return {
                success:
                    false,

                error:
                    error.message
            };
        }
    }
);

// ============================================================
// DEBUG INFORMATION
// ============================================================

console.log(
    "=================================================="
);

console.log(
    "AI PERSON PHOTO FINDER MAIN PROCESS LOADED"
);

console.log(
    "Project:",
    __dirname
);

console.log(
    "Incremental face-index workflow enabled."
);

console.log(
    "ZIP extraction cache enabled."
);

console.log(
    "Search script receives face_index.json directly."
);

console.log(
    "=================================================="
);
// ============================================================
// HIGH-SPEED GPU FOLDER SCAN IPC HANDLER
// ============================================================
ipcMain.handle("scan-folder-fast", async (event, folderPath) => {
    try {
        const resolvedPath = normalizePath(folderPath);
        if (!isDirectory(resolvedPath)) {
            throw new Error("Invalid folder path provided.");
        }

        const results = await gpuScanner.scanDirectory(resolvedPath, (current, total) => {
            if (mainWindow && !mainWindow.isDestroyed()) {
                mainWindow.webContents.send("scan-progress", { current, total });
            }
        });

        return {
            success: true,
            totalScanned: results.length,
            embeddings: results
        };
    } catch (error) {
        console.error("Fast GPU Scan Error:", error);
        return {
            success: false,
            error: error.message
        };
    }
});