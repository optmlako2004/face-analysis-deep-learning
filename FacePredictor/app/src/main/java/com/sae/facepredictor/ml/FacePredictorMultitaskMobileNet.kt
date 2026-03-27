package com.sae.facepredictor.ml

import android.content.Context
import android.graphics.Bitmap
import com.sae.facepredictor.data.model.Ethnicity
import com.sae.facepredictor.data.model.Gender
import com.sae.facepredictor.data.model.PredictionResult
import com.sae.facepredictor.utils.LogCapture
import com.sae.facepredictor.utils.estimateAgeConfidence
import org.tensorflow.lite.Interpreter
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel
import java.io.FileInputStream

/**
 * FacePredictorMultitaskMobileNet - Modèle multitâche basé sur MobileNetV3Small
 * Fichier: multitask_model_v5_mobilenet.tflite
 *
 * Sorties:
 * - age_output: Régression (0-116 ans)
 * - gender_output: Classification binaire (Homme/Femme)
 * - ethnicity_output: Classification multi-classe (4 classes)
 *
 * Plus léger (3.7 MB vs 11 MB) et plus rapide que le modèle EfficientNet
 */
class FacePredictorMultitaskMobileNet(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var initError: String? = null

    // Model input size (128x128)
    private val inputSize = 128

    // Ethnicity classes (4 classes)
    private val ethnicityClasses = arrayOf("Blanc", "Noir", "Asiatique", "Indien")

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            LogCapture.d(TAG, "Starting to load multitask MobileNet V5 model...")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }

            interpreter = Interpreter(loadModelFile("multitask_model_v5_mobilenet.tflite"), options)
            LogCapture.i(TAG, "Multitask MobileNet V5 model loaded successfully")
        } catch (e: Exception) {
            initError = e.message
            LogCapture.e(TAG, "Error loading multitask MobileNet model: ${e.message}", e)
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
        LogCapture.d(TAG, "Starting Multitask MobileNet prediction on bitmap ${bitmap.width}x${bitmap.height}")

        // Check if model is loaded
        if (interpreter == null) {
            LogCapture.e(TAG, "Multitask MobileNet model not loaded. Init error: $initError")
            return null
        }

        return try {
            // Resize image to 128x128
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            LogCapture.d(TAG, "Bitmap resized to ${inputSize}x${inputSize}")

            // Prepare input buffer
            val inputBuffer = bitmapToByteBuffer(resizedBitmap)

            // Prepare output buffers for multi-output model
            // Output order: gender (1), age (1), ethnicity (4)
            val genderOutput = Array(1) { FloatArray(1) }
            val ageOutput = Array(1) { FloatArray(1) }
            val ethnicityOutput = Array(1) { FloatArray(4) }

            // Map outputs to their indices
            val outputMap = HashMap<Int, Any>()
            outputMap[0] = genderOutput    // Index 0 = gender (sigmoid)
            outputMap[1] = ageOutput       // Index 1 = age (regression)
            outputMap[2] = ethnicityOutput // Index 2 = ethnicity (softmax)

            // Run inference with multiple outputs
            LogCapture.d(TAG, "Running multitask MobileNet inference...")
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
            val ethnicity = ethnicityFromIndex(maxIdx)
            val ethnicityConfidence = probs[maxIdx]
            LogCapture.d(TAG, "Ethnicity predicted: ${ethnicity.label} (confidence: $ethnicityConfidence)")

            LogCapture.i(TAG, "Multitask MobileNet Result: Age=$age, Gender=${gender.label}, Ethnicity=${ethnicity.label}")

            PredictionResult(
                age = age,
                ageConfidence = estimateAgeConfidence(age),
                gender = gender,
                genderConfidence = genderConfidence,
                ethnicity = ethnicity,
                ethnicityConfidence = ethnicityConfidence
            )
        } catch (e: Exception) {
            LogCapture.e(TAG, "Multitask MobileNet prediction failed: ${e.message}", e)
            null
        }
    }

    private fun ethnicityFromIndex(index: Int): Ethnicity {
        return when (index) {
            0 -> Ethnicity.WHITE
            1 -> Ethnicity.BLACK
            2 -> Ethnicity.ASIAN
            3 -> Ethnicity.INDIAN
            else -> Ethnicity.WHITE
        }
    }

    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
            // Extract RGB values (0-255 range as per model_info.json)
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
            LogCapture.d(TAG, "Multitask MobileNet interpreter closed")
        } catch (e: Exception) {
            LogCapture.e(TAG, "Error closing Multitask MobileNet interpreter", e)
        }
    }

    companion object {
        private const val TAG = "FacePredictorMultitaskMN"

        fun getModelInfo() = FacePredictorModel.ModelInfo(
            name = "Face Predictor V5 MobileNet",
            type = "Multi-task MobileNetV3Small",
            inputSize = "128x128x3",
            accuracy = "Léger & Rapide (3.7 MB)"
        )

        fun getAgeModelInfo() = FacePredictorModel.ModelInfo(
            name = "Age Predictor (V5 MobileNet)",
            type = "Multi-task MobileNetV3Small",
            inputSize = "128x128x3",
            accuracy = "MAE: 7.87 ans"
        )

        fun getGenderModelInfo() = FacePredictorModel.ModelInfo(
            name = "Gender Classifier (V5 MobileNet)",
            type = "Multi-task MobileNetV3Small",
            inputSize = "128x128x3",
            accuracy = "Accuracy: 77.9%"
        )

        fun getEthnicityModelInfo() = FacePredictorModel.ModelInfo(
            name = "Ethnicity Classifier (V5 MobileNet)",
            type = "Multi-task MobileNetV3Small (4 classes)",
            inputSize = "128x128x3",
            accuracy = "Accuracy: 59.2%"
        )
    }
}
