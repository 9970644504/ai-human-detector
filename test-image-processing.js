const path = require("path");
const fs = require("fs");

const {
    processImage
} = require("./services/imageService");


async function test() {

    const inputPath = path.join(
        __dirname,
        "test-image.jpg"
    );

    const outputDirectory = path.join(
        __dirname,
        "processed"
    );

    if (!fs.existsSync(outputDirectory)) {
        fs.mkdirSync(
            outputDirectory,
            {
                recursive: true
            }
        );
    }

    const outputPath = path.join(
        outputDirectory,
        "ai-ready.jpg"
    );

    try {

        console.log(
            "\nStarting image processing...\n"
        );

        const result = await processImage(
            inputPath,
            outputPath
        );

        console.log(
            "Original Image:"
        );

        console.log(result.original);

        console.log(
            "\nAI Image:"
        );

        console.log(result.processed);

        console.log(
            "\nImage processing successful!"
        );

    } catch (error) {

        console.error(
            "\nImage processing failed:"
        );

        console.error(
            error.message
        );
    }
}


test();