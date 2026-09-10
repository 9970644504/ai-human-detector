const {
    contextBridge,
    ipcRenderer
} = require("electron");


contextBridge.exposeInMainWorld(
    "electronAPI",
    {

        // ====================================================
        // REFERENCE PHOTO
        // ====================================================

        selectReference: () => {

            return ipcRenderer.invoke(
                "select-reference"
            );

        },


        // ====================================================
        // CAMERA
        // ====================================================

        startCamera: () => {

            return true;

        },

        saveCameraPhoto: (
            dataUrl
        ) => {

            return ipcRenderer.invoke(
                "save-camera-photo",
                dataUrl
            );

        },

        saveCameraSnapshot: (
            dataUrl
        ) => {

            return ipcRenderer.invoke(
                "save-camera-photo",
                dataUrl
            );

        },


        // ====================================================
        // FOLDER
        // ====================================================

        selectFolder: () => {

            return ipcRenderer.invoke(
                "select-folder"
            );

        },


        // ====================================================
        // ZIP
        // ====================================================

        selectZip: () => {

            return ipcRenderer.invoke(
                "select-zip"
            );

        },


        // ====================================================
        // COUNT IMAGES
        // ====================================================

        countImages: (
            sourcePath
        ) => {

            return ipcRenderer.invoke(
                "count-images",
                sourcePath
            );

        },


        // ====================================================
        // GET IMAGE FILES
        // ====================================================

        getImageFiles: (
            sourcePath
        ) => {

            return ipcRenderer.invoke(
                "get-image-files",
                sourcePath
            );

        },


        // ====================================================
        // RUN FACE SCAN
        // ====================================================

        runFaceScan: (
            referencePath,
            sourcePath
        ) => {

            return ipcRenderer.invoke(
                "run-face-scan",
                referencePath,
                sourcePath
            );

        },


        // ====================================================
        // GET RESULTS
        // ====================================================

        getScanResults: () => {

            return ipcRenderer.invoke(
                "get-scan-results"
            );

        },


        // ====================================================
        // PROGRESS
        // ====================================================

        onScanProgress: (
            callback
        ) => {

            if (
                typeof callback !== "function"
            ) {

                return;

            }

            const listener =
                (
                    event,
                    data
                ) => {

                    callback(
                        data
                    );

                };

            ipcRenderer.on(
                "scan-progress",
                listener
            );

            return () => {

                ipcRenderer.removeListener(
                    "scan-progress",
                    listener
                );

            };

        }

    }
);