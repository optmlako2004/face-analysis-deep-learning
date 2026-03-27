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
 * FacePredictorMobileNet - Utilise 3 modèles MobileNetV3 séparés pour les prédictions:
 * - gender_v3_mobilenet.tflite: Classification binaire (Homme/Femme)
 * - age_v3_mobilenet.tflite: Régression (0-116 ans)
 * - ethnicity_v3_mobilenet.tflite: Classification multi-classe (4 classes)
 *
 * Plus légers et rapides que les modèles EfficientNet V2
 */
class FacePredictorMobileNet(private val context: Context) {

    private var genderInterpreter: Interpreter? = null
    private var ageInterpreter: Interpreter? = null
    private var ethnicityInterpreter: Interpreter? = null
    private var initError: String? = null

    // Model input size (128x128 for all MobileNet models)
    private val inputSize = 128

    // Gender threshold
    private val genderThreshold = 0.5f

    // Ethnicity classes (4 classes)
    private val ethnicityClasses = arrayOf("Blanc", "Noir", "Asiatique", "Indien")

    init {
        loadModels()
    }

    private fun loadModels() {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }

            // Load Gender MobileNet model
            LogCapture.d(TAG, "Loading gender_v3_mobilenet model...")
            genderInterpreter = Interpreter(loadModelFile("gender_v3_mobilenet.tflite"), options)
            LogCapture.i(TAG, "Gender MobileNet model loaded")

            // Load Age MobileNet model
            LogCapture.d(TAG, "Loading age_v3_mobilenet model...")
            ageInterpreter = Interpreter(loadModelFile("age_v3_mobilenet.tflite"), options)
            LogCapture.i(TAG, "Age MobileNet model loaded")

            // Load Ethnicity MobileNet model
            LogCapture.d(TAG, "Loading ethnicity_v3_mobilenet model...")
            ethnicityInterpreter = Interpreter(loadModelFile("ethnicity_v3_mobilenet.tflite"), options)
            LogCapture.i(TAG, "Ethnicity MobileNet model loaded")

            LogCapture.i(TAG, "All MobileNet V3 models loaded successfully")
        } catch (e: Exception) {
            initError = e.message
            LogCapture.e(TAG, "Error loading MobileNet models: ${e.message}", e)
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
        LogCapture.d(TAG, "Starting MobileNet prediction on bitmap ${bitmap.width}x${bitmap.height}")

        // Check if models are loaded
        if (genderInterpreter == null || ageInterpreter == null || ethnicityInterpreter == null) {
            LogCapture.e(TAG, "MobileNet models not loaded. Init error: $initError")
            return null
        }

        return try {
            // Resize image to 128x128
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            LogCapture.d(TAG, "Bitmap resized to ${inputSize}x${inputSize}")

            // Prepare input buffer (RGB values 0-255)
            val inputBuffer = bitmapToByteBuffer(resizedBitmap)

            // === Gender Prediction ===
            val genderOutput = Array(1) { FloatArray(1) }
            inputBuffer.rewind()
            genderInterpreter!!.run(inputBuffer, genderOutput)

            val genderProb = genderOutput[0][0]
            val gender = if (genderProb > genderThreshold) Gender.FEMALE else Gender.MALE
            val genderConfidence = if (genderProb > genderThreshold) genderProb else (1f - genderProb)
            LogCapture.d(TAG, "Gender MobileNet: ${gender.label} (prob: $genderProb, conf: $genderConfidence)")

            // === Age Prediction ===
            val ageOutput = Array(1) { FloatArray(1) }
            inputBuffer.rewind()
            ageInterpreter!!.run(inputBuffer, ageOutput)

            val age = ageOutput[0][0].toInt().coerceIn(0, 116)
            LogCapture.d(TAG, "Age MobileNet: $age ans")

            // === Ethnicity Prediction ===
            val ethnicityOutput = Array(1) { FloatArray(4) } // 4 classes
            inputBuffer.rewind()
            ethnicityInterpreter!!.run(inputBuffer, ethnicityOutput)

            val probs = ethnicityOutput[0]
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val ethnicity = ethnicityFromIndex(maxIdx)
            val ethnicityConfidence = probs[maxIdx]
            LogCapture.d(TAG, "Ethnicity MobileNet: ${ethnicity.label} (conf: $ethnicityConfidence)")

            LogCapture.i(TAG, "MobileNet Result: Age=$age, Gender=${gender.label}, Ethnicity=${ethnicity.label}")

            PredictionResult(
                age = age,
                ageConfidence = estimateAgeConfidence(age),
                gender = gender,
                genderConfidence = genderConfidence,
                ethnicity = ethnicity,
                ethnicityConfidence = ethnicityConfidence
            )
        } catch (e: Exception) {
            LogCapture.e(TAG, "MobileNet Prediction failed: ${e.message}", e)
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
            genderInterpreter?.close()
            ageInterpreter?.close()
            ethnicityInterpreter?.close()
            LogCapture.d(TAG, "All MobileNet model interpreters closed")
        } catch (e: Exception) {
            LogCapture.e(TAG, "Error closing MobileNet interpreters", e)
        }
    }

    companion object {
        private const val TAG = "FacePredictorMobileNet"

        fun getGenderModelInfo() = FacePredictorModelV2.ModelInfo(
            name = "Gender Classifier V3",
            type = "MobileNetV3Small",
            inputSize = "128x128x3",
            accuracy = "Accuracy: ~78%"
        )

        fun getAgeModelInfo() = FacePredictorModelV2.ModelInfo(
            name = "Age Predictor V3",
            type = "MobileNetV3Small + CBAM",
            inputSize = "128x128x3",
            accuracy = "MAE: ~8 ans"
        )

        fun getEthnicityModelInfo() = FacePredictorModelV2.ModelInfo(
            name = "Ethnicity Classifier V3",
            type = "MobileNetV3Small (4 classes)",
            inputSize = "128x128x3",
            accuracy = "Accuracy: ~60%"
        )
    }
}
