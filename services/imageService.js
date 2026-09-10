const sharp = require("sharp");
const fs = require("fs");
const path = require("path");

const MAX_AI_WIDTH = 1280;
const MAX_AI_HEIGHT = 1280;


/**
 * Get information about an image
 */
async function getImageInfo(filePath) {

    if (!fs.existsSync(filePath)) {
        throw new Error("Image file does not exist.");
    }

    const stats = fs.statSync(filePath);

    const metadata = await sharp(filePath).metadata();

    return {
        fileName: path.basename(filePath),
        fileSize: stats.size,
        width: metadata.width || 0,
        height: metadata.height || 0,
        format: metadata.format || "unknown",
        orientation: metadata.orientation || null
    };
}


/**
 * Create an AI-ready image
 *
 * The original image is NOT modified.
 */
async function createAIImage(inputPath, outputPath) {

    await sharp(inputPath)
        .rotate()
        .resize({
            width: MAX_AI_WIDTH,
            height: MAX_AI_HEIGHT,
            fit: "inside",
            withoutEnlargement: true
        })
        .jpeg({
            quality: 90
        })
        .toFile(outputPath);

    return outputPath;
}


/**
 * Process the complete image
 */
async function processImage(inputPath, outputPath) {

    const originalInfo = await getImageInfo(inputPath);

    await createAIImage(
        inputPath,
        outputPath
    );

    const processedInfo = await getImageInfo(
        outputPath
    );

    return {
        original: originalInfo,
        processed: processedInfo
    };
}


/*
 * Export functions
 */
module.exports = {
    getImageInfo,
    createAIImage,
    processImage
};