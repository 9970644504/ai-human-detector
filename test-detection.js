const path = require("path");

const detectionService =
    require("./services/detectionService");


async function test() {

    const imagePath =
        path.join(
            __dirname,
            "test-image.jpg"
        );


    try {

        console.log(
            "\n=============================="
        );

        console.log(
            "AI HUMAN DETECTOR TEST"
        );

        console.log(
            "==============================\n"
        );


        /*
            Run YOLO
        */

        const output =
            await detectionService.detect(
                imagePath
            );


        console.log(
            "\nModel output:"
        );

        console.log(
            output
        );


        console.log(
            "\nTensor dimensions:"
        );

        console.log(
            output.dims
        );


        console.log(
            "\nAI test completed successfully!"
        );


    } catch (error) {

        console.error(
            "\nAI detection failed:"
        );

        console.error(
            error
        );

    }

}


test();