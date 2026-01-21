package com.sae.facepredictor.ml

import android.content.Context
import android.graphics.Bitmap
import com.sae.facepredictor.data.model.Ethnicity
import com.sae.facepredictor.data.model.Gender
import com.sae.facepredictor.data.model.PredictionResult
import com.sae.facepredictor.utils.LogCapture
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream

class FacePredictorModel(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var initError: String? = null

    // Model input size (128x128 as per model_info.json)
    private val inputSize = 128

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            LogCapture.d(TAG, "Starting to load multitask model V4...")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }

            interpreter = Interpreter(loadModelFile("multitask_model.tflite"), options)
            LogCapture.i(TAG, "Multitask model V4 loaded successfully")
        } catch (e: Exception) {
            initError = e.message
            LogCapture.e(TAG, "Error loading model: ${e.message}", e)
        }
    }

    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val assetFileDescriptor = context.assets.openFd(modelName)
        val fileInputStream = FileInputStream(assetFileDescriptor.fileDescriptor)
        val fileChannel = fileInputStream.channel
        val startOffset = assetFileDescriptor.startOffset
        val declaredLength = assetFileDescriptor.declaredLength
        val mappedBuffer = fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength)
        fileInputStream.close()
        return mappedBuffer
    }

    fun predict(bitmap: Bitmap): PredictionResult? {
        LogCapture.d(TAG, "Starting prediction on bitmap ${bitmap.width}x${bitmap.height}")

        // Check if model is loaded
        if (interpreter == null) {
            LogCapture.e(TAG, "Model not loaded. Init error: $initError")
            return null
        }

        return try {
            // Resize image to 128x128
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            LogCapture.d(TAG, "Bitmap resized to ${inputSize}x${inputSize}")

            // Prepare input buffer
            val inputBuffer = bitmapToByteBuffer(resizedBitmap)

            // Prepare output buffers for multi-output model
            // Output order (from TFLite): gender (1), age (1), ethnicity (4)
            val genderOutput = Array(1) { FloatArray(1) }
            val ageOutput = Array(1) { FloatArray(1) }
            val ethnicityOutput = Array(1) { FloatArray(4) }  // 4 classes: Blanc, Noir, Asiatique, Indien

            // Map outputs to their indices (order determined by TFLite conversion)
            val outputMap = HashMap<Int, Any>()
            outputMap[0] = genderOutput    // Index 0 = gender (sigmoid)
            outputMap[1] = ageOutput       // Index 1 = age (regression)
            outputMap[2] = ethnicityOutput // Index 2 = ethnicity (softmax)

            // Run inference with multiple outputs
            LogCapture.d(TAG, "Running multitask inference...")
            interpreter!!.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

            // Process age output
            val age = ageOutput[0][0].toInt().coerceIn(0, 116)
            LogCapture.d(TAG, "Age predicted: $age")

            // Process gender output (sigmoid, >0.5 = Female)
            val genderProb = genderOutput[0][0]
            val gender = if (genderProb > 0.5f) Gender.FEMALE else Gender.MALE
            val genderConfidence = if (genderProb > 0.5f) genderProb else (1f - genderProb)
            LogCapture.d(TAG, "Gender predicted: ${gender.label} (prob: $genderProb, confidence: $genderConfidence)")

            // Process ethnicity output (softmax)
            val probs = ethnicityOutput[0]
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val ethnicity = Ethnicity.fromIndex(maxIdx)
            val ethnicityConfidence = probs[maxIdx]
            LogCapture.d(TAG, "Ethnicity predicted: ${ethnicity.label} (confidence: $ethnicityConfidence)")

            PredictionResult(
                age = age,
                ageConfidence = 0.85f,
                gender = gender,
                genderConfidence = genderConfidence,
                ethnicity = ethnicity,
                ethnicityConfidence = ethnicityConfidence
            )
        } catch (e: Exception) {
            LogCapture.e(TAG, "Prediction failed: ${e.message}", e)
            null
        }
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            // Extract RGB values (Android ARGB format)
            // Keep in [0, 255] range as per model_info.json
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()

            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    fun close() {
        try {
            interpreter?.close()
            LogCapture.d(TAG, "Model interpreter closed")
        } catch (e: Exception) {
            LogCapture.e(TAG, "Error closing interpreter", e)
        }
    }

    data class ModelInfo(
        val name: String,
        val type: String,
        val inputSize: String,
        val accuracy: String
    )

    companion object {
        private const val TAG = "FacePredictorModel"

        // Metrics from V4 model (balanced dataset, 4 ethnicity classes)
        fun getAgeModelInfo() = ModelInfo(
            name = "Age Predictor (V4)",
            type = "Multi-task CNN Regression",
            inputSize = "128x128x3",
            accuracy = "MAE: 6.65 ans"
        )

        fun getGenderModelInfo() = ModelInfo(
            name = "Gender Classifier (V4)",
            type = "Multi-task Binary Classification",
            inputSize = "128x128x3",
            accuracy = "Precision: 78.1%"
        )

        fun getEthnicityModelInfo() = ModelInfo(
            name = "Ethnicity Classifier (V4)",
            type = "Multi-task Multi-class (4 classes)",
            inputSize = "128x128x3",
            accuracy = "Precision: 64.3%"
        )

        fun getMultitaskModelInfo() = ModelInfo(
            name = "Face Predictor V4",
            type = "Multi-task EfficientNetB0",
            inputSize = "128x128x3",
            accuracy = "Dataset equilibre"
        )
    }
}
