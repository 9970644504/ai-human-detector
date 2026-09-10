// ============================================================
// AI PERSON PHOTO FINDER
// RENDERER
// ============================================================

"use strict";

// ============================================================
// STATE
// ============================================================

let referencePath = null;
let photoSourcePath = null;
let photoSourceType = null;

let currentResults = [];
let currentUncertain = [];

let scanRunning = false;

// ============================================================
// DOM HELPERS
// ============================================================

function $(id) {
    return document.getElementById(id);
}

// ============================================================
// ELEMENTS
// ============================================================

const selectionPage = $("selectionPage");
const resultsPage = $("resultsPage");

const referenceUpload = $("referenceUpload");
const referencePreview = $("referencePreview");
const referenceButton = $("referenceButton");
const referenceFileName = $("referenceFileName");

const folderUpload = $("folderUpload");
const folderPreview = $("folderPreview");
const folderButton = $("folderButton");
const zipButton = $("zipButton");
const folderName = $("folderName");
const photoCount = $("photoCount");

const findButton = $("findButton");
const searchStatus = $("searchStatus");
const backButton = $("backButton");

const statusDot = $("statusDot");
const headerStatus = $("headerStatus");

const resultStatus = $("resultStatus");
const matchingCountTop = $("matchingCountTop");

const progressPercentage = $("progressPercentage");
const progressFill = $("progressFill");
const progressDetails = $("progressDetails");

const resultsContainer = $("resultsContainer");
const resultsEmpty = $("resultsEmpty");

const scannedCount = $("scannedCount");
const matchingCount = $("matchingCount");
const uncertainCount = $("uncertainCount");

const resultText = $("resultText");

const uncertainSection = $("uncertainSection");
const uncertainBadge = $("uncertainBadge");
const uncertainList = $("uncertainList");

// ============================================================
// INITIALIZE
// ============================================================

document.addEventListener("DOMContentLoaded", () => {
    setupEvents();
    updateFindButton();

    setHeaderStatus(
        "System Ready",
        "ready"
    );
});

// ============================================================
// EVENTS
// ============================================================

function setupEvents() {

    // --------------------------------------------------------
    // Reference photo
    // --------------------------------------------------------

    if (referenceButton) {
        referenceButton.addEventListener(
            "click",
            selectReferencePhoto
        );
    }

    if (referenceUpload) {
        referenceUpload.addEventListener(
            "click",
            selectReferencePhoto
        );
    }

    // --------------------------------------------------------
    // Camera Captured Photo Event Listener
    // --------------------------------------------------------

    window.addEventListener("reference-photo-selected", (event) => {
        const capturedPath = event.detail.path;
        if (!capturedPath) return;

        // 1. Assign the captured camera path to referencePath
        referencePath = capturedPath;

        // 2. Update UI elements using your renderer helpers
        if (referenceFileName) {
            referenceFileName.textContent = getFileName(capturedPath);
        }

        showReferencePreview(capturedPath);

        if (searchStatus) {
            searchStatus.textContent =
                "Camera photo captured. Now select a photo collection.";
        }

        // 3. Enable the Start Search button if collection is selected
        updateFindButton();

        setHeaderStatus(
            "Reference captured",
            "ready"
        );
    });

    // --------------------------------------------------------
    // Folder
    // --------------------------------------------------------

    if (folderButton) {
        folderButton.addEventListener(
            "click",
            selectFolder
        );
    }

    if (folderUpload) {
        folderUpload.addEventListener(
            "click",
            selectFolder
        );
    }

    // --------------------------------------------------------
    // ZIP
    // --------------------------------------------------------

    if (zipButton) {
        zipButton.addEventListener(
            "click",
            selectZip
        );
    }

    // --------------------------------------------------------
    // Search
    // --------------------------------------------------------

    if (findButton) {
        findButton.addEventListener(
            "click",
            startSearch
        );
    }

    // --------------------------------------------------------
    // Back
    // --------------------------------------------------------

    if (backButton) {
        backButton.addEventListener(
            "click",
            showSelectionPage
        );
    }

    // --------------------------------------------------------
    // Scan progress
    // --------------------------------------------------------

    if (
        window.electronAPI &&
        typeof window.electronAPI.onScanProgress === "function"
    ) {
        window.electronAPI.onScanProgress(
            handleScanProgress
        );
    }
}

// ============================================================
// REFERENCE PHOTO
// ============================================================

async function selectReferencePhoto() {

    if (scanRunning) {
        return;
    }

    if (
        !window.electronAPI ||
        typeof window.electronAPI.selectReference !== "function"
    ) {
        showError(
            "Reference photo API is not available."
        );
        return;
    }

    try {

        const selected =
            await window.electronAPI.selectReference();

        if (!selected) {
            return;
        }

        referencePath = selected;

        referenceFileName.textContent =
            getFileName(selected);

        showReferencePreview(selected);

        searchStatus.textContent =
            "Reference photo selected. Now select a photo collection.";

        updateFindButton();

        setHeaderStatus(
            "Reference selected",
            "ready"
        );

    } catch (error) {

        console.error(
            "Reference selection error:",
            error
        );

        showError(
            error.message ||
            "Could not select reference photo."
        );
    }
}

// ============================================================
// REFERENCE PREVIEW
// ============================================================

function showReferencePreview(filePath) {

    if (!referencePreview) {
        return;
    }

    referencePreview.innerHTML = "";

    const image =
        document.createElement("img");

    image.src =
        toFileUrl(filePath);

    image.alt =
        "Reference photo";

    image.className =
        "reference-preview-image";

    image.onerror = () => {
        console.error(
            "Could not load reference image:",
            filePath
        );
    };

    referencePreview.appendChild(
        image
    );
}

// ============================================================
// FOLDER
// ============================================================

async function selectFolder() {

    if (scanRunning) {
        return;
    }

    if (
        !window.electronAPI ||
        typeof window.electronAPI.selectFolder !== "function"
    ) {
        showError(
            "Folder selection API is not available."
        );
        return;
    }

    try {

        const selected =
            await window.electronAPI.selectFolder();

        if (!selected) {
            return;
        }

        photoSourcePath = selected;
        photoSourceType = "folder";

        folderName.textContent =
            selected;

        folderPreview.innerHTML = `
            <div class="upload-icon">📁</div>
            <div class="upload-title">Folder Selected</div>
            <div class="upload-description">
                Counting photos...
            </div>
        `;

        await updatePhotoCount();

        searchStatus.textContent =
            "Photo collection selected. Ready to search.";

        updateFindButton();

        setHeaderStatus(
            "Collection selected",
            "ready"
        );

    } catch (error) {

        console.error(
            "Folder selection error:",
            error
        );

        showError(
            error.message ||
            "Could not select folder."
        );
    }
}

// ============================================================
// ZIP
// ============================================================

async function selectZip() {

    if (scanRunning) {
        return;
    }

    if (
        !window.electronAPI ||
        typeof window.electronAPI.selectZip !== "function"
    ) {
        showError(
            "ZIP selection API is not available."
        );
        return;
    }

    try {

        const selected =
            await window.electronAPI.selectZip();

        if (!selected) {
            return;
        }

        photoSourcePath = selected;
        photoSourceType = "zip";

        folderName.textContent =
            selected;

        folderPreview.innerHTML = `
            <div class="upload-icon">🗜️</div>
            <div class="upload-title">ZIP Selected</div>
            <div class="upload-description">
                Counting photos...
            </div>
        `;

        await updatePhotoCount();

        searchStatus.textContent =
            "ZIP collection selected. Ready to search.";

        updateFindButton();

        setHeaderStatus(
            "ZIP collection selected",
            "ready"
        );

    } catch (error) {

        console.error(
            "ZIP selection error:",
            error
        );

        showError(
            error.message ||
            "Could not select ZIP file."
        );
    }
}

// ============================================================
// COUNT PHOTOS
// ============================================================

async function updatePhotoCount() {

    if (!photoSourcePath) {

        if (photoCount) {
            photoCount.textContent = "0";
        }

        return;
    }

    if (photoCount) {
        photoCount.textContent = "...";
    }

    if (
        !window.electronAPI ||
        typeof window.electronAPI.countImages !== "function"
    ) {
        showError(
            "Image counting API is not available."
        );
        return;
    }

    try {

        const count =
            await window.electronAPI.countImages(
                photoSourcePath
            );

        const safeCount =
            Number.isFinite(Number(count))
                ? Number(count)
                : 0;

        if (photoCount) {
            photoCount.textContent =
                String(safeCount);
        }

        if (safeCount === 0) {

            folderPreview.innerHTML = `
                <div class="upload-icon">⚠️</div>

                <div class="upload-title">
                    No Photos Found
                </div>

                <div class="upload-description">
                    Supported: JPG, PNG, WEBP, BMP, GIF
                </div>
            `;

        } else {

            const sourceLabel =
                photoSourceType === "zip"
                    ? "ZIP collection ready"
                    : "Collection Ready";

            const icon =
                photoSourceType === "zip"
                    ? "🗜️"
                    : "📁";

            folderPreview.innerHTML = `
                <div class="upload-icon">${icon}</div>

                <div class="upload-title">
                    ${sourceLabel}
                </div>

                <div class="upload-description">
                    ${safeCount}
                    photo${safeCount === 1 ? "" : "s"} found
                </div>
            `;
        }

    } catch (error) {

        console.error(
            "Count images error:",
            error
        );

        if (photoCount) {
            photoCount.textContent = "0";
        }

        showError(
            error.message ||
            "Could not count photos."
        );
    }
}

// ============================================================
// FIND BUTTON
// ============================================================

function updateFindButton() {

    if (!findButton) {
        return;
    }

    findButton.disabled =
        scanRunning ||
        !referencePath ||
        !photoSourcePath;
}

// ============================================================
// START SEARCH
// ============================================================

async function startSearch() {

    if (
        scanRunning ||
        !referencePath ||
        !photoSourcePath
    ) {
        return;
    }

    if (
        !window.electronAPI ||
        typeof window.electronAPI.runFaceScan !== "function"
    ) {
        setErrorState(
            "Face scan API is not available."
        );
        return;
    }

    scanRunning = true;

    updateFindButton();

    resetResults();

    showResultsPage();

    setScanState();

    setProgress(
        0,
        "Starting AI scanner..."
    );

    try {

        console.log(
            "[SCAN] Reference:",
            referencePath
        );

        console.log(
            "[SCAN] Source:",
            photoSourcePath
        );

        console.log(
            "[SCAN] Source type:",
            photoSourceType
        );

        // IMPORTANT:
        // Folder and ZIP both use the same API.
        // main.js handles ZIP extraction.
        const response =
            await window.electronAPI.runFaceScan(
                referencePath,
                photoSourcePath
            );

        console.log(
            "[SCAN] Response:",
            response
        );

        if (
            !response ||
            response.success !== true
        ) {

            throw new Error(
                response?.error ||
                "Face scan failed."
            );
        }

        const data =
            response.data || {};

        displayResults(data);

        finishScan();

    } catch (error) {

        console.error(
            "Search error:",
            error
        );

        setErrorState(
            error.message ||
            "Face scan failed."
        );

    } finally {

        scanRunning = false;

        updateFindButton();
    }
}

// ============================================================
// PROGRESS
// ============================================================

function handleScanProgress(data) {

    if (!data) {
        return;
    }

    console.log(
        "[SCAN PROGRESS]",
        data
    );

    const type =
        data.type || "info";

    const message =
        String(data.message || "");

    // --------------------------------------------------------
    // Explicit numeric progress
    // --------------------------------------------------------

    if (
        typeof data.percent === "number" &&
        Number.isFinite(data.percent)
    ) {

        setProgress(
            data.percent,
            message
        );

        // Do not return here if there is additional
        // useful scanning state.
        if (type === "scanning") {

            resultStatus.textContent =
                "Scanning";

            resultStatus.className =
                "detection-status status-scanning";

            setHeaderStatus(
                "Searching...",
                "scanning"
            );
        }

        return;
    }

    // --------------------------------------------------------
    // Scanning
    // --------------------------------------------------------

    if (type === "scanning") {

        resultStatus.textContent =
            "Scanning";

        resultStatus.className =
            "detection-status status-scanning";

        resultText.textContent =
            "Scanning";

        setHeaderStatus(
            "Searching...",
            "scanning"
        );

        if (message) {

            progressDetails.textContent =
                cleanProgressMessage(message);
        }

        return;
    }

    // --------------------------------------------------------
    // Complete
    // --------------------------------------------------------

    if (type === "complete") {

        setProgress(
            100,
            message ||
            "Scan complete."
        );

        return;
    }

    // --------------------------------------------------------
    // Error
    // --------------------------------------------------------

    if (
        type === "error" ||
        type === "python-error"
    ) {

        if (message) {

            progressDetails.textContent =
                cleanProgressMessage(message);
        }

        return;
    }

    // --------------------------------------------------------
    // Normal information
    // --------------------------------------------------------

    if (message) {

        progressDetails.textContent =
            cleanProgressMessage(message);
    }
}

// ============================================================
// PROGRESS BAR
// ============================================================

function setProgress(percent, message) {

    let value =
        Number(percent);

    if (!Number.isFinite(value)) {
        value = 0;
    }

    value =
        Math.max(
            0,
            Math.min(
                100,
                value
            )
        );

    if (progressFill) {

        progressFill.style.width =
            `${value}%`;
    }

    if (progressPercentage) {

        progressPercentage.textContent =
            `${Math.round(value)}%`;
    }

    if (
        message &&
        progressDetails
    ) {

        progressDetails.textContent =
            cleanProgressMessage(message);
    }
}

// ============================================================
// CLEAN PROGRESS MESSAGE
// ============================================================

function cleanProgressMessage(message) {

    if (!message) {
        return "";
    }

    const text =
        String(message).trim();

    // Python JSON progress
    if (
        text.startsWith("PROGRESS_JSON:")
    ) {
        return "Processing photos...";
    }

    // Python JSON result
    if (
        text.startsWith("RESULT_JSON:")
    ) {
        return "Finishing scan...";
    }

    // Common scanner messages
    if (
        /^Scanning:/i.test(text)
    ) {
        return text;
    }

    if (
        /^Loading/i.test(text)
    ) {
        return text;
    }

    if (
        /^Creating reference/i.test(text)
    ) {
        return text;
    }

    if (
        /^Processing/i.test(text)
    ) {
        return text;
    }

    if (
        /^Comparing/i.test(text)
    ) {
        return text;
    }

    return text;
}

// ============================================================
// DISPLAY RESULTS
// ============================================================

function displayResults(data) {

    if (!data) {
        data = {};
    }

    const summary =
        data.summary || {};

    const results =
        Array.isArray(data.results)
            ? data.results
            : [];

    // --------------------------------------------------------
    // MATCHES
    // --------------------------------------------------------

    currentResults =
        results.filter(
            item =>
                item &&
                String(item.status).toUpperCase() === "MATCH"
        );

    // --------------------------------------------------------
    // UNCERTAIN
    // --------------------------------------------------------

    currentUncertain =
        results.filter(
            item =>
                item &&
                String(item.status).toUpperCase() === "UNCERTAIN"
        );

    // --------------------------------------------------------
    // COUNTS
    // --------------------------------------------------------

    const scanned =
        Number(
            summary.images_processed ??
            summary.images_found ??
            data.images_processed ??
            data.images_found ??
            results.length ??
            0
        );

    const matches =
        Number(
            summary.matches ??
            currentResults.length
        );

    const uncertain =
        Number(
            summary.uncertain ??
            currentUncertain.length
        );

    // --------------------------------------------------------
    // UPDATE UI
    // --------------------------------------------------------

    scannedCount.textContent =
        String(
            Number.isFinite(scanned)
                ? scanned
                : 0
        );

    matchingCount.textContent =
        String(
            Number.isFinite(matches)
                ? matches
                : currentResults.length
        );

    matchingCountTop.textContent =
        String(
            Number.isFinite(matches)
                ? matches
                : currentResults.length
        );

    uncertainCount.textContent =
        String(
            Number.isFinite(uncertain)
                ? uncertain
                : currentUncertain.length
        );

    resultText.textContent =
        "Complete";

    // --------------------------------------------------------
    // RENDER
    // --------------------------------------------------------

    renderMatchResults(
        currentResults
    );

    renderUncertainResults(
        currentUncertain
    );
}

// ============================================================
// MATCH RESULTS
// ============================================================

function renderMatchResults(results) {

    if (!resultsContainer) {
        return;
    }

    resultsContainer.innerHTML = "";

    if (
        !Array.isArray(results) ||
        results.length === 0
    ) {

        resultsContainer.appendChild(
            createEmptyElement(
                "🔎",
                "No matching photos",
                "No photos crossed the match threshold."
            )
        );

        return;
    }

    results.forEach(
        (result, index) => {

            const card =
                createResultCard(
                    result,
                    index
                );

            resultsContainer.appendChild(
                card
            );
        }
    );
}

// ============================================================
// CREATE RESULT CARD
// ============================================================

function createResultCard(result, index) {

    const card =
        document.createElement("article");

    card.className =
        "result-card";

    // --------------------------------------------------------
    // IMAGE
    // --------------------------------------------------------

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "result-image-wrapper";

    const image =
        document.createElement("img");

    image.className =
        "result-image";

    image.alt =
        getFileName(result?.image);

    image.src =
        toFileUrl(result?.image);

    image.loading =
        "lazy";

    image.onerror = () => {

        image.style.display =
            "none";
    };

    // --------------------------------------------------------
    // REMOVE BUTTON
    // --------------------------------------------------------

    const removeButton =
        document.createElement("button");

    removeButton.type =
        "button";

    removeButton.className =
        "remove-result-button";

    removeButton.title =
        "Remove from results";

    removeButton.setAttribute(
        "aria-label",
        "Remove result"
    );

    removeButton.textContent =
        "×";

    removeButton.addEventListener(
        "click",
        event => {

            event.preventDefault();
            event.stopPropagation();

            removeMatch(index);
        }
    );

    wrapper.appendChild(
        image
    );

    wrapper.appendChild(
        removeButton
    );

    // --------------------------------------------------------
    // INFO
    // --------------------------------------------------------

    const info =
        document.createElement("div");

    info.className =
        "result-info";

    const status =
        document.createElement("div");

    status.className =
        "result-status match";

    status.textContent =
        "MATCH";

    const score =
        document.createElement("div");

    score.className =
        "result-score";

    score.textContent =
        `Similarity: ${formatScore(
            result?.best_similarity
        )}`;

    const file =
        document.createElement("div");

    file.className =
        "result-file";

    file.title =
        result?.image || "";

    file.textContent =
        getFileName(result?.image);

    info.appendChild(
        status
    );

    info.appendChild(
        score
    );

    info.appendChild(
        file
    );

    card.appendChild(
        wrapper
    );

    card.appendChild(
        info
    );

    return card;
}

// ============================================================
// REMOVE MATCH
// ============================================================

function removeMatch(index) {

    if (
        index < 0 ||
        index >= currentResults.length
    ) {
        return;
    }

    currentResults.splice(
        index,
        1
    );

    renderMatchResults(
        currentResults
    );

    matchingCountTop.textContent =
        String(
            currentResults.length
        );

    matchingCount.textContent =
        String(
            currentResults.length
        );
}

// ============================================================
// UNCERTAIN RESULTS
// ============================================================

function renderUncertainResults(results) {

    if (
        !uncertainSection ||
        !uncertainList ||
        !uncertainBadge
    ) {
        return;
    }

    uncertainList.innerHTML =
        "";

    if (
        !Array.isArray(results) ||
        results.length === 0
    ) {

        uncertainSection.classList.add(
            "hidden"
        );

        uncertainBadge.textContent =
            "0";

        return;
    }

    uncertainSection.classList.remove(
        "hidden"
    );

    uncertainBadge.textContent =
        String(
            results.length
        );

    results.forEach(
        result => {

            const card =
                createUncertainCard(
                    result
                );

            uncertainList.appendChild(
                card
            );
        }
    );
}

// ============================================================
// UNCERTAIN CARD
// ============================================================

function createUncertainCard(result) {

    const card =
        document.createElement("article");

    card.className =
        "result-card";

    // --------------------------------------------------------
    // IMAGE
    // --------------------------------------------------------

    const wrapper =
        document.createElement("div");

    wrapper.className =
        "result-image-wrapper";

    const image =
        document.createElement("img");

    image.className =
        "result-image";

    image.src =
        toFileUrl(result?.image);

    image.alt =
        getFileName(result?.image);

    image.loading =
        "lazy";

    image.onerror = () => {

        image.style.display =
            "none";
    };

    // --------------------------------------------------------
    // INFO
    // --------------------------------------------------------

    const info =
        document.createElement("div");

    info.className =
        "result-info";

    const status =
        document.createElement("div");

    status.className =
        "result-status uncertain";

    status.textContent =
        "UNCERTAIN";

    const score =
        document.createElement("div");

    score.className =
        "result-score";

    score.textContent =
        `Similarity: ${formatScore(
            result?.best_similarity
        )}`;

    const file =
        document.createElement("div");

    file.className =
        "result-file";

    file.title =
        result?.image || "";

    file.textContent =
        getFileName(result?.image);

    wrapper.appendChild(
        image
    );

    info.appendChild(
        status
    );

    info.appendChild(
        score
    );

    info.appendChild(
        file
    );

    card.appendChild(
        wrapper
    );

    card.appendChild(
        info
    );

    return card;
}

// ============================================================
// EMPTY ELEMENT
// ============================================================

function createEmptyElement(
    icon,
    title,
    message
) {

    const element =
        document.createElement("div");

    element.className =
        "empty-results";

    element.innerHTML = `
        <div class="empty-results-icon">
            ${escapeHtml(icon)}
        </div>

        <h3>
            ${escapeHtml(title)}
        </h3>

        <p>
            ${escapeHtml(message)}
        </p>
    `;

    return element;
}

// ============================================================
// SCAN STATE
// ============================================================

function setScanState() {

    resultStatus.textContent =
        "Scanning";

    resultStatus.className =
        "detection-status status-scanning";

    resultText.textContent =
        "Scanning";

    setHeaderStatus(
        "Searching...",
        "scanning"
    );
}

// ============================================================
// FINISH
// ============================================================

function finishScan() {

    setProgress(
        100,
        "Scan complete."
    );

    resultStatus.textContent =
        "Complete";

    resultStatus.className =
        "detection-status status-complete";

    resultText.textContent =
        "Complete";

    setHeaderStatus(
        "System Ready",
        "ready"
    );
}

// ============================================================
// ERROR STATE
// ============================================================

function setErrorState(message) {

    resultStatus.textContent =
        "Error";

    resultStatus.className =
        "detection-status status-error";

    resultText.textContent =
        "Error";

    progressDetails.textContent =
        message;

    setHeaderStatus(
        "Scan Error",
        "error"
    );

    scanRunning =
        false;

    updateFindButton();

    alert(message);
}

// ============================================================
// ERROR
// ============================================================

function showError(message) {

    console.error(
        message
    );

    alert(
        message
    );
}

// ============================================================
// SHOW RESULTS
// ============================================================

function showResultsPage() {

    selectionPage.classList.add(
        "hidden"
    );

    resultsPage.classList.remove(
        "hidden"
    );

    window.scrollTo(
        0,
        0
    );
}

// ============================================================
// SHOW SELECTION
// ============================================================

function showSelectionPage() {

    if (scanRunning) {

        const leave =
            confirm(
                "A scan is currently running. Go back anyway?"
            );

        if (!leave) {
            return;
        }
    }

    selectionPage.classList.remove(
        "hidden"
    );

    resultsPage.classList.add(
        "hidden"
    );

    window.scrollTo(
        0,
        0
    );
}

// ============================================================
// RESET RESULTS
// ============================================================

function resetResults() {

    currentResults = [];
    currentUncertain = [];

    resultsContainer.innerHTML = "";

    resultsContainer.appendChild(
        createEmptyElement(
            "🔎",
            "Scanning photos...",
            "AI is detecting faces and comparing embeddings."
        )
    );

    uncertainList.innerHTML =
        "";

    uncertainSection.classList.add(
        "hidden"
    );

    uncertainBadge.textContent =
        "0";

    matchingCountTop.textContent =
        "0";

    scannedCount.textContent =
        "0";

    matchingCount.textContent =
        "0";

    uncertainCount.textContent =
        "0";

    resultText.textContent =
        "Scanning";

    setProgress(
        0,
        "Starting AI scanner..."
    );
}

// ============================================================
// HEADER STATUS
// ============================================================

function setHeaderStatus(
    message,
    state
) {

    if (headerStatus) {
        headerStatus.textContent =
            message;
    }

    if (!statusDot) {
        return;
    }

    statusDot.className =
        "status-dot";

    if (state === "scanning") {

        statusDot.classList.add(
            "scanning"
        );
    }

    if (state === "error") {

        statusDot.classList.add(
            "error"
        );
    }
}

// ============================================================
// FILE NAME
// ============================================================

function getFileName(filePath) {

    if (!filePath) {
        return "Unknown file";
    }

    const normalized =
        String(filePath)
            .replace(/\\/g, "/");

    return (
        normalized
            .split("/")
            .pop() ||
        "Unknown file"
    );
}

// ============================================================
// FILE URL
// ============================================================

function toFileUrl(filePath) {

    if (!filePath) {
        return "";
    }

    let normalized =
        String(filePath)
            .replace(/\\/g, "/");

    // Already a file URL
    if (
        normalized.startsWith("file:///")
    ) {
        return normalized;
    }

    // Windows path
    // Example:
    // C:/Photos/person.jpg
    if (
        /^[A-Za-z]:\//.test(normalized)
    ) {

        return (
            "file:///" +
            encodeURI(normalized)
        );
    }

    // Linux / macOS
    if (
        normalized.startsWith("/")
    ) {

        return (
            "file://" +
            encodeURI(normalized)
        );
    }

    return encodeURI(
        normalized
    );
}

// ============================================================
// SCORE
// ============================================================

function formatScore(score) {

    const number =
        Number(score);

    if (
        !Number.isFinite(number)
    ) {
        return "N/A";
    }

    return number.toFixed(4);
}

// ============================================================
// ESCAPE HTML
// ============================================================

function escapeHtml(value) {

    return String(
        value ?? ""
    )
        .replace(
            /&/g,
            "&amp;"
        )
        .replace(
            /</g,
            "&lt;"
        )
        .replace(
            />/g,
            "&gt;"
        )
        .replace(
            /"/g,
            "&quot;"
        )
        .replace(
            /'/g,
            "&#039;"
        );
}
// ... your existing renderer.js code above ...

