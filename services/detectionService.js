const ort = require("onnxruntime-node");
const sharp = require("sharp");
const path = require("path");

const INPUT_SIZE = 640;

const PERSON_CLASS_ID = 0;

const CONFIDENCE_THRESHOLD = 0.35;


/*
    Detection Service
*/

class DetectionService {

    constructor(modelPath) {

        this.modelPath = modelPath;

        this.session = null;

        this.inputName = null;

        this.outputName = null;
    }


    /*
        Load YOLO model
    */

    async loadModel() {

        if (this.session) {

            return;

        }


        console.log(
            "Loading YOLO model..."
        );


        this.session =
            await ort.InferenceSession.create(
                this.modelPath
            );


        this.inputName =
            this.session.inputNames[0];


        this.outputName =
            this.session.outputNames[0];


        console.log(
            "YOLO model loaded successfully."
        );

    }


    /*
        Get original image information
    */

    async getImageInfo(imagePath) {

        const metadata =
            await sharp(imagePath)
                .metadata();


        return {

            width: metadata.width,

            height: metadata.height,

            format: metadata.format

        };

    }


    /*
        Letterbox preprocessing

        Keeps the original aspect ratio.
    */

    async imageToTensor(imagePath) {

        const metadata =
            await sharp(imagePath)
                .metadata();


        const originalWidth =
            metadata.width;


        const originalHeight =
            metadata.height;


        /*
            Calculate scale.

            The image will fit inside
            640 × 640.
        */

        const scale =
            Math.min(
                INPUT_SIZE / originalWidth,
                INPUT_SIZE / originalHeight
            );


        const resizedWidth =
            Math.round(
                originalWidth * scale
            );


        const resizedHeight =
            Math.round(
                originalHeight * scale
            );


        /*
            Padding
        */

        const padX =
            Math.floor(
                (INPUT_SIZE - resizedWidth) / 2
            );


        const padY =
            Math.floor(
                (INPUT_SIZE - resizedHeight) / 2
            );


        /*
            Resize image while preserving
            aspect ratio.

            Then place it on a 640×640
            canvas.
        */

        const imageBuffer =
            await sharp(imagePath)

                .resize({
                    width: resizedWidth,
                    height: resizedHeight,
                    fit: "fill"
                })

                .extend({

                    top: padY,

                    bottom:
                        INPUT_SIZE -
                        resizedHeight -
                        padY,

                    left: padX,

                    right:
                        INPUT_SIZE -
                        resizedWidth -
                        padX,

                    background: {
                        r: 114,
                        g: 114,
                        b: 114,
                        alpha: 1
                    }

                })

                .removeAlpha()

                .raw()

                .toBuffer();


        /*
            Convert RGB → CHW
        */

        const inputData =
            new Float32Array(
                3 *
                INPUT_SIZE *
                INPUT_SIZE
            );


        const pixelCount =
            INPUT_SIZE *
            INPUT_SIZE;


        for (
            let i = 0;
            i < pixelCount;
            i++
        ) {

            inputData[i] =
                imageBuffer[
                    i * 3
                ] / 255.0;


            inputData[
                pixelCount + i
            ] =
                imageBuffer[
                    i * 3 + 1
                ] / 255.0;


            inputData[
                pixelCount * 2 + i
            ] =
                imageBuffer[
                    i * 3 + 2
                ] / 255.0;

        }


        return {

            tensor:
                new ort.Tensor(
                    "float32",
                    inputData,
                    [
                        1,
                        3,
                        INPUT_SIZE,
                        INPUT_SIZE
                    ]
                ),

            originalWidth,

            originalHeight,

            scale,

            padX,

            padY,

            resizedWidth,

            resizedHeight
        };

    }


    /*
        Convert YOLO coordinates
        back to original image.
    */

    parseDetections(
        output,
        preprocessing
    ) {

        const {

            originalWidth,

            originalHeight,

            scale,

            padX,

            padY

        } = preprocessing;


        const detections = [];


        const data =
            output.data;


        const numberOfDetections =
            output.dims[1];


        for (
            let i = 0;
            i < numberOfDetections;
            i++
        ) {

            const offset =
                i * 6;


            const x1 =
                data[offset];


            const y1 =
                data[offset + 1];


            const x2 =
                data[offset + 2];


            const y2 =
                data[offset + 3];


            const confidence =
                data[offset + 4];


            const classId =
                Math.round(
                    data[offset + 5]
                );


            /*
                Only person
            */

            if (
                classId !==
                PERSON_CLASS_ID
            ) {

                continue;

            }


            /*
                Confidence filter
            */

            if (
                confidence <
                CONFIDENCE_THRESHOLD
            ) {

                continue;

            }


            /*
                Remove letterbox padding
            */

            let originalX1 =
                (x1 - padX) / scale;


            let originalY1 =
                (y1 - padY) / scale;


            let originalX2 =
                (x2 - padX) / scale;


            let originalY2 =
                (y2 - padY) / scale;


            /*
                Keep coordinates
                inside image.
            */

            originalX1 =
                Math.max(
                    0,
                    Math.min(
                        originalWidth,
                        originalX1
                    )
                );


            originalY1 =
                Math.max(
                    0,
                    Math.min(
                        originalHeight,
                        originalY1
                    )
                );


            originalX2 =
                Math.max(
                    0,
                    Math.min(
                        originalWidth,
                        originalX2
                    )
                );


            originalY2 =
                Math.max(
                    0,
                    Math.min(
                        originalHeight,
                        originalY2
                    )
                );


            /*
                Ignore invalid boxes
            */

            if (
                originalX2 <=
                originalX1 ||
                originalY2 <=
                originalY1
            ) {

                continue;

            }


            detections.push({

                classId,

                className:
                    "person",

                confidence:
                    Number(
                        confidence.toFixed(4)
                    ),

                confidencePercent:
                    Number(
                        (
                            confidence *
                            100
                        ).toFixed(2)
                    ),

                x:
                    Math.round(
                        originalX1
                    ),

                y:
                    Math.round(
                        originalY1
                    ),

                width:
                    Math.round(
                        originalX2 -
                        originalX1
                    ),

                height:
                    Math.round(
                        originalY2 -
                        originalY1
                    ),

                x1:
                    Math.round(
                        originalX1
                    ),

                y1:
                    Math.round(
                        originalY1
                    ),

                x2:
                    Math.round(
                        originalX2
                    ),

                y2:
                    Math.round(
                        originalY2
                    )

            });

        }


        return detections;

    }


    /*
        Main detection function
    */

    async detect(imagePath) {

        await this.loadModel();


        const preprocessing =
            await this.imageToTensor(
                imagePath
            );


        const feeds = {

            [this.inputName]:
                preprocessing.tensor

        };


        console.log(
            "Running AI detection..."
        );


        const results =
            await this.session.run(
                feeds
            );


        const output =
            results[
                this.outputName
            ];


        console.log(
            "AI inference completed."
        );


        const detections =
            this.parseDetections(
                output,
                preprocessing
            );


        return {

            image: {

                width:
                    preprocessing.originalWidth,

                height:
                    preprocessing.originalHeight

            },

            peopleCount:
                detections.length,

            detections

        };

    }

}


/*
    Model location
*/

const modelPath =
    path.join(
        __dirname,
        "..",
        "models",
        "yolo26n.onnx"
    );


/*
    Create service
*/

const detectionService =
    new DetectionService(
        modelPath
    );


module.exports =
    detectionService;