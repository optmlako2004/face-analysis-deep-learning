package com.sae.facepredictor.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Matrix
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
import kotlin.math.max
import kotlin.math.min

/**
 * FacePredictorHybrid - Combine le meilleur des deux approches:
 * - Genre: gender_v2_model.tflite (V2 Orienté - meilleur pour le genre)
 * - Age: Ensemble V2 + V4 avec TTA (flip horizontal) pour meilleure précision
 * - Ethnicité: multitask_model.tflite (V4 Multitâche - meilleur pour l'ethnicité)
 */
class FacePredictorHybrid(private val context: Context) {

    // V2 models for gender and age
    private var genderInterpreter: Interpreter? = null
    private var ageInterpreter: Interpreter? = null

    // V4 multitask model for ethnicity
    private var multitaskInterpreter: Interpreter? = null

    private var initError: String? = null

    private val inputSize = 128
    private val genderThreshold = 0.5f

    // Ethnicity classes V4 (4 classes, no Others)
    private val ethnicityClassesV4 = arrayOf("Blanc", "Noir", "Asiatique", "Indien")

    init {
        loadModels()
    }

    private fun loadModels() {
        try {
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }

            // Load Gender V2 model (best for gender)
            LogCapture.d(TAG, "Loading gender_v2 model for hybrid...")
            genderInterpreter = Interpreter(loadModelFile("gender_v2_model.tflite"), options)
            LogCapture.i(TAG, "Gender V2 model loaded")

            // Load Age V2 model
            LogCapture.d(TAG, "Loading age_v2 model for hybrid...")
            ageInterpreter = Interpreter(loadModelFile("age_v2_model.tflite"), options)
            LogCapture.i(TAG, "Age V2 model loaded")

            // Load Multitask V4 model (best for ethnicity)
            LogCapture.d(TAG, "Loading multitask_model for hybrid ethnicity...")
            multitaskInterpreter = Interpreter(loadModelFile("multitask_model.tflite"), options)
            LogCapture.i(TAG, "Multitask V4 model loaded")

            LogCapture.i(TAG, "Hybrid model loaded successfully (Gender V2 + Age V2 + Ethnicity V4)")
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

        if (genderInterpreter == null || ageInterpreter == null || multitaskInterpreter == null) {
            LogCapture.e(TAG, "Models not loaded. Init error: $initError")
            return null
        }

        return try {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            LogCapture.d(TAG, "Bitmap resized to ${inputSize}x${inputSize}")

            // Prepare input buffers (original)
            val inputBuffer = bitmapToByteBuffer(resizedBitmap)
            val inputBufferCLAHE = bitmapToByteBufferWithCLAHE(resizedBitmap)

            // Prepare flipped input buffer (TTA)
            val flippedBitmap = flipHorizontally(resizedBitmap)
            val inputBufferFlipped = bitmapToByteBuffer(flippedBitmap)

            // === Gender from V2 (best for gender) ===
            val genderOutput = Array(1) { FloatArray(1) }
            inputBufferCLAHE.rewind()
            genderInterpreter!!.run(inputBufferCLAHE, genderOutput)

            val genderProb = genderOutput[0][0]
            val gender = if (genderProb > genderThreshold) Gender.FEMALE else Gender.MALE
            val genderConfidence = if (genderProb > genderThreshold) genderProb else (1f - genderProb)
            LogCapture.d(TAG, "Gender (V2): ${gender.label} (conf: ${genderConfidence * 100}%)")

            // === Age: Ensemble V2 + V4 with TTA (4 predictions averaged) ===

            // Age V2 on original
            val ageOutput = Array(1) { FloatArray(1) }
            inputBuffer.rewind()
            ageInterpreter!!.run(inputBuffer, ageOutput)
            val ageV2 = ageOutput[0][0]

            // Age V2 on flipped (TTA)
            val ageOutputFlipped = Array(1) { FloatArray(1) }
            inputBufferFlipped.rewind()
            ageInterpreter!!.run(inputBufferFlipped, ageOutputFlipped)
            val ageV2Flip = ageOutputFlipped[0][0]

            // V4 Multitask on original (also gives ethnicity)
            val genderOutV4 = Array(1) { FloatArray(1) }
            val ageOutV4 = Array(1) { FloatArray(1) }
            val ethnicityOutV4 = Array(1) { FloatArray(4) }

            val outputMap = HashMap<Int, Any>()
            outputMap[0] = genderOutV4
            outputMap[1] = ageOutV4
            outputMap[2] = ethnicityOutV4

            inputBuffer.rewind()
            multitaskInterpreter!!.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)
            val ageV4 = ageOutV4[0][0]

            // V4 Multitask on flipped (TTA)
            val genderOutV4Flip = Array(1) { FloatArray(1) }
            val ageOutV4Flip = Array(1) { FloatArray(1) }
            val ethnicityOutV4Flip = Array(1) { FloatArray(4) }

            val outputMapFlip = HashMap<Int, Any>()
            outputMapFlip[0] = genderOutV4Flip
            outputMapFlip[1] = ageOutV4Flip
            outputMapFlip[2] = ethnicityOutV4Flip

            inputBufferFlipped.rewind()
            multitaskInterpreter!!.runForMultipleInputsOutputs(arrayOf(inputBufferFlipped), outputMapFlip)
            val ageV4Flip = ageOutV4Flip[0][0]

            // Ensemble + TTA: average of 4 predictions
            val ageEnsemble = (ageV2 + ageV2Flip + ageV4 + ageV4Flip) / 4f
            val age = ageEnsemble.toInt().coerceIn(0, 116)
            LogCapture.d(TAG, "Age Ensemble+TTA: V2=${"%.1f".format(ageV2)}, V2flip=${"%.1f".format(ageV2Flip)}, V4=${"%.1f".format(ageV4)}, V4flip=${"%.1f".format(ageV4Flip)} => $age ans")

            // === Ethnicity from V4 Multitask (average original + flipped) ===
            val ethProbs = ethnicityOutV4[0]
            val ethProbsFlip = ethnicityOutV4Flip[0]
            val ethProbsAvg = FloatArray(4) { (ethProbs[it] + ethProbsFlip[it]) / 2f }
            val maxIdx = ethProbsAvg.indices.maxByOrNull { ethProbsAvg[it] } ?: 0
            val ethnicity = ethnicityFromIndexV4(maxIdx)
            val ethnicityConfidence = ethProbsAvg[maxIdx]
            LogCapture.d(TAG, "Ethnicity (V4 TTA): ${ethnicity.label} (conf: ${ethnicityConfidence * 100}%)")

            LogCapture.i(TAG, "HYBRID Result: Age=$age, Gender=${gender.label}, Ethnicity=${ethnicity.label}")

            PredictionResult(
                age = age,
                ageConfidence = 0.85f,
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

    private fun ethnicityFromIndexV4(index: Int): Ethnicity {
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

    private fun bitmapToByteBufferWithCLAHE(bitmap: Bitmap): ByteBuffer {
        val byteBuffer = ByteBuffer.allocateDirect(4 * inputSize * inputSize * 3)
        byteBuffer.order(ByteOrder.nativeOrder())

        val pixels = IntArray(inputSize * inputSize)
        bitmap.getPixels(pixels, 0, inputSize, 0, 0, inputSize, inputSize)

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
            multitaskInterpreter?.close()
            LogCapture.d(TAG, "Hybrid model interpreters closed")
        } catch (e: Exception) {
            LogCapture.e(TAG, "Error closing interpreters", e)
        }
    }

    companion object {
        private const val TAG = "FacePredictorHybrid"
    }
}
