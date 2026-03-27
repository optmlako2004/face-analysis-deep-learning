package com.sae.facepredictor.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
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
 * FacePredictorHybrid - Combine le meilleur de chaque modele specialise:
 * - Genre: gender_v2_model.tflite (modele specialise EfficientNetB0)
 * - Age: age_v2_model.tflite avec TTA (flip horizontal)
 * - Ethnicite: ethnicity_v2_model.tflite (modele specialise EfficientNetB0)
 *
 * Tous les modeles sont des EfficientNetB0 re-entraines, input 224x224.
 */
class FacePredictorHybrid(private val context: Context) {

    private var genderInterpreter: Interpreter? = null
    private var ageInterpreter: Interpreter? = null
    private var ethnicityInterpreter: Interpreter? = null

    private var initError: String? = null

    private val inputSize = 224
    private val genderThreshold = 0.5f

    init {
        loadModels()
    }

    private fun loadModels() {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }

            LogCapture.d(TAG, "Loading gender_v2 model for hybrid...")
            genderInterpreter = Interpreter(loadModelFile("gender_v2_model.tflite"), options)
            LogCapture.i(TAG, "Gender V2 model loaded")

            LogCapture.d(TAG, "Loading age_v2 model for hybrid...")
            ageInterpreter = Interpreter(loadModelFile("age_v2_model.tflite"), options)
            LogCapture.i(TAG, "Age V2 model loaded")

            LogCapture.d(TAG, "Loading ethnicity_v2 model for hybrid...")
            ethnicityInterpreter = Interpreter(loadModelFile("ethnicity_v2_model.tflite"), options)
            LogCapture.i(TAG, "Ethnicity V2 model loaded")

            LogCapture.i(TAG, "Hybrid model loaded (Genre V2 + Age V2 TTA + Ethnicite V2)")
        } catch (e: Exception) {
            initError = e.message
            LogCapture.e(TAG, "Error loading hybrid models: ${e.message}", e)
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
        LogCapture.d(TAG, "Starting HYBRID prediction on bitmap ${bitmap.width}x${bitmap.height}")

        if (genderInterpreter == null || ageInterpreter == null || ethnicityInterpreter == null) {
            LogCapture.e(TAG, "Models not loaded. Init error: $initError")
            return null
        }

        return try {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

            // Prepare input buffers (original + flipped for TTA)
            val inputBuffer = bitmapToByteBuffer(resizedBitmap)
            val flippedBitmap = flipHorizontally(resizedBitmap)
            val inputBufferFlipped = bitmapToByteBuffer(flippedBitmap)

            // === Genre from V2 specialise ===
            val genderOutput = Array(1) { FloatArray(1) }
            inputBuffer.rewind()
            genderInterpreter!!.run(inputBuffer, genderOutput)

            val genderProb = genderOutput[0][0]
            val gender = if (genderProb > genderThreshold) Gender.FEMALE else Gender.MALE
            val genderConfidence = if (genderProb > genderThreshold) genderProb else (1f - genderProb)
            LogCapture.d(TAG, "Gender (V2): ${gender.label} (conf: ${genderConfidence * 100}%)")

            // === Age: V2 avec TTA (original + flipped, moyenne) ===
            val ageOutput = Array(1) { FloatArray(1) }
            inputBuffer.rewind()
            ageInterpreter!!.run(inputBuffer, ageOutput)
            val ageOriginal = ageOutput[0][0]

            val ageOutputFlipped = Array(1) { FloatArray(1) }
            inputBufferFlipped.rewind()
            ageInterpreter!!.run(inputBufferFlipped, ageOutputFlipped)
            val ageFlipped = ageOutputFlipped[0][0]

            val ageAvg = (ageOriginal + ageFlipped) / 2f
            val age = ageAvg.toInt().coerceIn(0, 116)
            LogCapture.d(TAG, "Age TTA: original=${"%.1f".format(ageOriginal)}, flipped=${"%.1f".format(ageFlipped)} => $age ans")

            // === Ethnicite: V2 specialise avec TTA ===
            val ethOutput = Array(1) { FloatArray(5) }
            inputBuffer.rewind()
            ethnicityInterpreter!!.run(inputBuffer, ethOutput)

            val ethOutputFlipped = Array(1) { FloatArray(5) }
            inputBufferFlipped.rewind()
            ethnicityInterpreter!!.run(inputBufferFlipped, ethOutputFlipped)

            val ethProbsAvg = FloatArray(5) { (ethOutput[0][it] + ethOutputFlipped[0][it]) / 2f }
            val maxIdx = ethProbsAvg.indices.maxByOrNull { ethProbsAvg[it] } ?: 0
            val ethnicity = ethnicityFromIndex(maxIdx)
            val ethnicityConfidence = ethProbsAvg[maxIdx]
            LogCapture.d(TAG, "Ethnicity TTA: ${ethnicity.label} (conf: ${ethnicityConfidence * 100}%)")

            LogCapture.i(TAG, "HYBRID Result: Age=$age, Gender=${gender.label}, Ethnicity=${ethnicity.label}")

            PredictionResult(
                age = age,
                ageConfidence = estimateAgeConfidence(age),
                gender = gender,
                genderConfidence = genderConfidence,
                ethnicity = ethnicity,
                ethnicityConfidence = ethnicityConfidence
            )
        } catch (e: Exception) {
            LogCapture.e(TAG, "Hybrid prediction failed: ${e.message}", e)
            null
        }
    }

    private fun flipHorizontally(bitmap: Bitmap): Bitmap {
        val matrix = Matrix()
        matrix.postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun ethnicityFromIndex(index: Int): Ethnicity {
        return when (index) {
            0 -> Ethnicity.WHITE
            1 -> Ethnicity.BLACK
            2 -> Ethnicity.ASIAN
            3 -> Ethnicity.INDIAN
            4 -> Ethnicity.OTHERS
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
            LogCapture.d(TAG, "Hybrid model interpreters closed")
        } catch (e: Exception) {
            LogCapture.e(TAG, "Error closing interpreters", e)
        }
    }

    companion object {
        private const val TAG = "FacePredictorHybrid"
    }
}