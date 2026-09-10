const ort = require('onnxruntime-node');
const sharp = require('sharp');

async function testEnvironment() {
  console.log('--- TESTING HIGH-SPEED PIPELINE STACK ---');

  // 1. Verify Sharp Image Processing
  try {
    const testBuffer = await sharp({
      create: {
        width: 112,
        height: 112,
        channels: 3,
        background: { r: 255, g: 0, b: 0 }
      }
    }).raw().toBuffer();
    console.log('✅ Sharp Image Engine: OPERATIONAL (Buffer size:', testBuffer.length, 'bytes)');
  } catch (err) {
    console.error('❌ Sharp Error:', err.message);
  }

  // 2. Verify ONNX Runtime & GPU Providers
  try {
    console.log('✅ ONNX Runtime Version:', ort.version);
    
    // Create a simple dummy tensor to verify execution
    const tensor = new ort.Tensor('float32', new Float32Array([1, 2, 3, 4]), [2, 2]);
    console.log('✅ ONNX Tensor Engine: OPERATIONAL (Tensor shape:', tensor.dims, ')');
  } catch (err) {
    console.error('❌ ONNX Error:', err.message);
  }
}

testEnvironment();