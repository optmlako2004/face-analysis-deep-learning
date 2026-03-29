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
import kotlin.math.max
import kotlin.math.min

/**
 * FacePredictorModelV2 - Utilise 3 modèles séparés pour les prédictions:
 * - gender_v2_model.tflite: Classification binaire (Homme/Femme)
 * - age_v2_model.tflite: Régression (0-116 ans)
 * - ethnicity_v2_model.tflite: Classification multi-classe (5 classes avec Others)
 */
class FacePredictorModelV2(private val context: Context) {

    private var genderInterpreter: Interpreter? = null
    private var ageInterpreter: Interpreter? = null
    private var ethnicityInterpreter: Interpreter? = null
    private var initError: String? = null

    // Model input size (224x224 for EfficientNetB0 models)
    private val inputSize = 224

    // Gender threshold (from training)
    private val genderThreshold = 0.5f

    // Ethnicity classes V2 (5 classes with Others)
    private val ethnicityClasses = arrayOf("White", "Black", "Asian", "Indian", "Others")
    private val ethnicityLabels = arrayOf("Blanc", "Noir", "Asiatique", "Indien", "Autre")

    init {
        loadModels()
    }

    private fun loadModels() {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }

            // Load Gender V2 model
            LogCapture.d(TAG, "Loading gender_v2 model...")
            genderInterpreter = Interpreter(loadModelFile("gender_v2_model.tflite"), options)
            LogCapture.i(TAG, "Gender V2 model loaded")

            // Load Age V2 model
            LogCapture.d(TAG, "Loading age_v2 model...")
            ageInterpreter = Interpreter(loadModelFile("age_v2_model.tflite"), options)
            LogCapture.i(TAG, "Age V2 model loaded")

            // Load Ethnicity V2 model
            LogCapture.d(TAG, "Loading ethnicity_v2 model...")
            ethnicityInterpreter = Interpreter(loadModelFile("ethnicity_v2_model.tflite"), options)
            LogCapture.i(TAG, "Ethnicity V2 model loaded")

            LogCapture.i(TAG, "All V2 models loaded successfully")
        } catch (e: Exception) {
            initError = e.message
            LogCapture.e(TAG, "Error loading models: ${e.message}", e)
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
        LogCapture.d(TAG, "Starting V2 prediction on bitmap ${bitmap.width}x${bitmap.height}")

        // Check if models are loaded
        if (genderInterpreter == null || ageInterpreter == null || ethnicityInterpreter == null) {
            LogCapture.e(TAG, "Models not loaded. Init error: $initError")
            return null
        }

        return try {
            // Resize image to 128x128
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            LogCapture.d(TAG, "Bitmap resized to ${inputSize}x${inputSize}")

            // Prepare input buffer (standard)
            val inputBuffer = bitmapToByteBuffer(resizedBitmap)

            // Prepare input buffer with CLAHE for gender model
            val inputBufferCLAHE = bitmapToByteBufferWithCLAHE(resizedBitmap)

            // === Gender Prediction ===
            val genderOutput = Array(1) { FloatArray(1) }
            inputBufferCLAHE.rewind()
            genderInterpreter!!.run(inputBufferCLAHE, genderOutput)

            val genderProb = genderOutput[0][0]
            val gender = if (genderProb > genderThreshold) Gender.FEMALE else Gender.MALE
            val genderConfidence = if (genderProb > genderThreshold) genderProb else (1f - genderProb)
            LogCapture.d(TAG, "Gender V2: ${gender.label} (prob: $genderProb, conf: $genderConfidence)")

            // === Age Prediction ===
            val ageOutput = Array(1) { FloatArray(1) }
            inputBuffer.rewind()
            ageInterpreter!!.run(inputBuffer, ageOutput)

            val age = ageOutput[0][0].toInt().coerceIn(0, 116)
            LogCapture.d(TAG, "Age V2: $age ans")

            // === Ethnicity Prediction ===
            val ethnicityOutput = Array(1) { FloatArray(5) } // 5 classes with Others
            inputBuffer.rewind()
            ethnicityInterpreter!!.run(inputBuffer, ethnicityOutput)

            val probs = ethnicityOutput[0]
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val ethnicity = ethnicityFromIndexV2(maxIdx)
            val ethnicityConfidence = probs[maxIdx]
            LogCapture.d(TAG, "Ethnicity V2: ${ethnicity.label} (conf: $ethnicityConfidence)")

            PredictionResult(
                age = age,
                ageConfidence = estimateAgeConfidence(age),
                gender = gender,
                genderConfidence = genderConfidence,
                ethnicity = ethnicity,
                ethnicityConfidence = ethnicityConfidence
            )
        } catch (e: Exception) {
            LogCapture.e(TAG, "V2 Prediction failed: ${e.message}", e)
            null
        }
    }

    /**
     * Convert ethnicity index to Ethnicity enum (V2 has 5 classes)
     */
    private fun ethnicityFromIndexV2(index: Int): Ethnicity {
        return when (index) {
            0 -> Ethnicity.WHITE
            1 -> Ethnicity.BLACK
            2 -> Ethnicity.ASIAN
            3 -> Ethnicity.INDIAN
            else -> Ethnicity.WHITE // "Others" maps to default
        }
    }

    /**
     * Standard bitmap to ByteBuffer conversion (RGB values 0-255)
     */
    private fun bitmapToByteBuffer(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        for (pixel in pixels) {
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

    /**
     * Bitmap to ByteBuffer with CLAHE-like enhancement (for gender model)
     * Simplified CLAHE: contrast enhancement using histogram equalization approximation
     */
    private fun bitmapToByteBufferWithCLAHE(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

        // Calculate min/max for contrast stretching (simple CLAHE approximation)
        var minL = 255f
        var maxL = 0f

        for (pixel in pixels) {
            val r = ((pixel shr 16) and 0xFF).toFloat()
            val g = ((pixel shr 8) and 0xFF).toFloat()
            val b = (pixel and 0xFF).toFloat()
            val l = 0.299f * r + 0.587f * g + 0.114f * b
            minL = min(minL, l)
            maxL = max(maxL, l)
        }

        val range = maxL - minL
        val scale = if (range > 0) 255f / range else 1f

        for (pixel in pixels) {
            var r = ((pixel shr 16) and 0xFF).toFloat()
            var g = ((pixel shr 8) and 0xFF).toFloat()
            var b = (pixel and 0xFF).toFloat()

            // Apply contrast enhancement
            if (range > 0) {
                r = ((r - minL) * scale).coerceIn(0f, 255f)
                g = ((g - minL) * scale).coerceIn(0f, 255f)
                b = ((b - minL) * scale).coerceIn(0f, 255f)
            }

            byteBuffer.putFloat(r)
            byteBuffer.putFloat(g)
            byteBuffer.putFloat(b)
        }

        byteBuffer.rewind()
        return byteBuffer
    }

    fun close() {
        try {
            genderInterpreter?.close()
            ageInterpreter?.close()
            ethnicityInterpreter?.close()
            LogCapture.d(TAG, "All V2 model interpreters closed")
        } catch (e: Exception) {
            LogCapture.e(TAG, "Error closing interpreters", e)
        }
    }

    companion object {
        private const val TAG = "FacePredictorModelV2"

        // Model info for V2 models (will be updated with actual metrics)
        fun getGenderModelInfo() = ModelInfo(
            name = "Gender Classifier V2",
            type = "EfficientNetB0 + CLAHE",
            inputSize = "128x128x3",
            accuracy = "Accuracy: ~90%"
        )

        fun getAgeModelInfo() = ModelInfo(
            name = "Age Predictor V2",
            type = "EfficientNetB0 Regression",
            inputSize = "128x128x3",
            accuracy = "MAE: ~6 ans"
        )

        fun getEthnicityModelInfo() = ModelInfo(
            name = "Ethnicity Classifier V2",
            type = "EfficientNetB0 (5 classes)",
            inputSize = "128x128x3",
            accuracy = "Accuracy: ~70%"
        )
    }

    data class ModelInfo(
        val name: String,
        val type: String,
        val inputSize: String,
        val accuracy: String
    )
}
