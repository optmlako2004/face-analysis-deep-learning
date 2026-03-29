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

/**
 * FacePredictorMoE - Mixture of Experts avec backbone MobileNetV3Small
 *
 * Architecture:
 *   MobileNetV3Small (backbone partage)
 *     -> 3 experts par tache (age, genre, ethnicite)
 *     -> gating network par tache (selectionne/pondere les experts)
 *
 * Modele unique multitache avec 3 sorties:
 *   - gender_output: sigmoid (1 float)
 *   - age_output: regression (1 float)
 *   - ethnicity_output: softmax (4 floats)
 */
class FacePredictorMoE(private val context: Context) {

    private var interpreter: Interpreter? = null
    private var initError: String? = null

    private val inputSize = 224

    init {
        loadModel()
    }

    private fun loadModel() {
        try {
            LogCapture.d(TAG, "Loading MoE MobileNetV3 model...")
            val options = Interpreter.Options().apply {
                setNumThreads(4)
            }
            interpreter = Interpreter(loadModelFile("moe_mobilenetv3.tflite"), options)

            // Log output tensor info for debugging
            val outputCount = interpreter!!.outputTensorCount
            LogCapture.i(TAG, "MoE model loaded: $outputCount outputs")
            for (i in 0 until outputCount) {
                val tensor = interpreter!!.getOutputTensor(i)
                LogCapture.d(TAG, "  Output[$i]: ${tensor.name()}, shape=${tensor.shape().contentToString()}")
            }
        } catch (e: Exception) {
            initError = e.message
            LogCapture.e(TAG, "Error loading MoE model: ${e.message}", e)
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

    /**
     * Raw inference on a single 128x128 bitmap (original + flip TTA).
     * Returns (ageRaw, genderProb, ethProbs[4]).
     */
    private fun inferTTA(resizedBitmap: Bitmap): Triple<Float, Float, FloatArray> {
        val flipMatrix = Matrix().apply {
            postScale(-1f, 1f, resizedBitmap.width / 2f, resizedBitmap.height / 2f)
        }
        val flippedBitmap = Bitmap.createBitmap(
            resizedBitmap, 0, 0, resizedBitmap.width, resizedBitmap.height, flipMatrix, true
        )

        // Inference on original
        val ageOut1 = Array(1) { FloatArray(1) }
        val genderOut1 = Array(1) { FloatArray(1) }
        val ethOut1 = Array(1) { FloatArray(4) }
        val outputMap1 = HashMap<Int, Any>().apply {
            put(0, ageOut1); put(1, genderOut1); put(2, ethOut1)
        }
        interpreter!!.runForMultipleInputsOutputs(arrayOf(bitmapToByteBuffer(resizedBitmap)), outputMap1)

        // Inference on flipped
        val ageOut2 = Array(1) { FloatArray(1) }
        val genderOut2 = Array(1) { FloatArray(1) }
        val ethOut2 = Array(1) { FloatArray(4) }
        val outputMap2 = HashMap<Int, Any>().apply {
            put(0, ageOut2); put(1, genderOut2); put(2, ethOut2)
        }
        interpreter!!.runForMultipleInputsOutputs(arrayOf(bitmapToByteBuffer(flippedBitmap)), outputMap2)

        val age = (ageOut1[0][0] + ageOut2[0][0]) / 2f
        val gender = (genderOut1[0][0] + genderOut2[0][0]) / 2f
        val eth = FloatArray(4) { i -> (ethOut1[0][i] + ethOut2[0][i]) / 2f }

        return Triple(age, gender, eth)
    }

    private fun buildResult(age: Float, genderProb: Float, ethProbs: FloatArray): PredictionResult {
        val gender = if (genderProb > 0.5f) Gender.FEMALE else Gender.MALE
        val genderConfidence = if (genderProb > 0.5f) genderProb else (1f - genderProb)
        val clampedAge = age.toInt().coerceIn(0, 116)
        val maxIdx = ethProbs.indices.maxByOrNull { ethProbs[it] } ?: 0
        val ethnicity = ethnicityFromIndex(maxIdx)
        val ethnicityConfidence = ethProbs[maxIdx]

        return PredictionResult(
            age = clampedAge,
            ageConfidence = estimateAgeConfidence(clampedAge),
            gender = gender,
            genderConfidence = genderConfidence,
            ethnicity = ethnicity,
            ethnicityConfidence = ethnicityConfidence
        )
    }

    /**
     * Standard prediction with flip TTA (used in real-time mode).
     */
    fun predict(bitmap: Bitmap): PredictionResult? {
        LogCapture.d(TAG, "Starting MoE prediction on bitmap ${bitmap.width}x${bitmap.height}")

        if (interpreter == null) {
            LogCapture.e(TAG, "MoE model not loaded. Init error: $initError")
            return null
        }

        return try {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)

            // Diagnostic: log original dimensions and center pixel RGB
            val cx = resizedBitmap.width / 2
            val cy = resizedBitmap.height / 2
            val centerPixel = resizedBitmap.getPixel(cx, cy)
            val cR = (centerPixel shr 16) and 0xFF
            val cG = (centerPixel shr 8) and 0xFF
            val cB = centerPixel and 0xFF
            LogCapture.d(TAG, "MoE input: original=${bitmap.width}x${bitmap.height}, resized=${inputSize}x${inputSize}, centerRGB=($cR,$cG,$cB)")

            LogCapture.d(TAG, "Running MoE inference (TTA)...")
            val (age, genderProb, ethProbs) = inferTTA(resizedBitmap)
            val result = buildResult(age, genderProb, ethProbs)

            LogCapture.i(TAG, "MoE TTA Result: Age=${result.age}, Gender=${result.gender.label}, Ethnicity=${result.ethnicity.label} (conf: ${result.ethnicityConfidence * 100}%)")
            result
        } catch (e: Exception) {
            LogCapture.e(TAG, "MoE prediction failed: ${e.message}", e)
            null
        }
    }

    /**
     * Enhanced prediction with multi-crop TTA for photo/import mode.
     * Creates 5 crop variants (simulates EMA effect from real-time):
     *   1. Original
     *   2. Center crop 95%
     *   3. Center crop 90%
     *   4. Slight shift left-up (3%)
     *   5. Slight shift right-down (3%)
     * Each variant gets flip TTA → 10 inferences total, averaged.
     */
    fun predictEnhanced(bitmap: Bitmap): PredictionResult? {
        LogCapture.d(TAG, "Starting MoE ENHANCED prediction on bitmap ${bitmap.width}x${bitmap.height}")

        if (interpreter == null) {
            LogCapture.e(TAG, "MoE model not loaded. Init error: $initError")
            return null
        }

        return try {
            val w = bitmap.width
            val h = bitmap.height

            // Generate crop variants from the face bitmap
            val crops = mutableListOf<Bitmap>()

            // 1. Original (full)
            crops.add(bitmap)

            // 2. Center crop 95%
            val m1 = (0.025f * minOf(w, h)).toInt().coerceAtLeast(1)
            if (w - 2 * m1 > 0 && h - 2 * m1 > 0) {
                crops.add(Bitmap.createBitmap(bitmap, m1, m1, w - 2 * m1, h - 2 * m1))
            }

            // 3. Center crop 90%
            val m2 = (0.05f * minOf(w, h)).toInt().coerceAtLeast(1)
            if (w - 2 * m2 > 0 && h - 2 * m2 > 0) {
                crops.add(Bitmap.createBitmap(bitmap, m2, m2, w - 2 * m2, h - 2 * m2))
            }

            // 4. Shift left-up 3%
            val s = (0.03f * minOf(w, h)).toInt().coerceAtLeast(1)
            if (w - s > 0 && h - s > 0) {
                crops.add(Bitmap.createBitmap(bitmap, 0, 0, w - s, h - s))
            }

            // 5. Shift right-down 3%
            if (s < w && s < h) {
                crops.add(Bitmap.createBitmap(bitmap, s, s, w - s, h - s))
            }

            LogCapture.d(TAG, "Running MoE enhanced inference (${crops.size} crops x 2 TTA = ${crops.size * 2} inferences)...")

            // Infer on all crops and accumulate
            var totalAge = 0f
            var totalGender = 0f
            val totalEth = FloatArray(4)
            var count = 0

            for (crop in crops) {
                val resized = Bitmap.createScaledBitmap(crop, inputSize, inputSize, true)
                val (age, gender, eth) = inferTTA(resized)
                totalAge += age
                totalGender += gender
                for (i in 0 until 4) totalEth[i] += eth[i]
                count++
            }

            val avgAge = totalAge / count
            val avgGender = totalGender / count
            val avgEth = FloatArray(4) { i -> totalEth[i] / count }

            val result = buildResult(avgAge, avgGender, avgEth)
            LogCapture.i(TAG, "MoE Enhanced Result ($count crops): Age=${result.age}, Gender=${result.gender.label}, Ethnicity=${result.ethnicity.label} (conf: ${result.ethnicityConfidence * 100}%)")
            result
        } catch (e: Exception) {
            LogCapture.e(TAG, "MoE enhanced prediction failed: ${e.message}", e)
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
            interpreter?.close()
            LogCapture.d(TAG, "MoE interpreter closed")
        } catch (e: Exception) {
            LogCapture.e(TAG, "Error closing MoE interpreter", e)
        }
    }

    companion object {
        private const val TAG = "FacePredictorMoE"
    }
}
