const path = require('path');
const ort = require('onnxruntime-node');

async function inspect() {
  const detectPath = path.join(__dirname, 'models/version-RFB-320.onnx');
  const recogPath = path.join(__dirname, 'models/face_recognition.onnx');

  const detSession = await ort.InferenceSession.create(detectPath);
  console.log('--- DETECTOR (version-RFB-320.onnx) ---');
  console.log('Input Names:', detSession.inputNames);
  console.log('Output Names:', detSession.outputNames);

  const recSession = await ort.InferenceSession.create(recogPath);
  console.log('\n--- RECOGNITION (face_recognition.onnx) ---');
  console.log('Input Names:', recSession.inputNames);
  console.log('Output Names:', recSession.outputNames);
}

inspect().catch(err => console.error(err));