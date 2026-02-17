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

    private val inputSize = 128

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

    fun predict(bitmap: Bitmap): PredictionResult? {
        LogCapture.d(TAG, "Starting MoE prediction on bitmap ${bitmap.width}x${bitmap.height}")

        if (interpreter == null) {
            LogCapture.e(TAG, "MoE model not loaded. Init error: $initError")
            return null
        }

        return try {
            val resizedBitmap = Bitmap.createScaledBitmap(bitmap, inputSize, inputSize, true)
            val inputBuffer = bitmapToByteBuffer(resizedBitmap)

            // Output buffers - TFLite reorders: output[0]=age(:1), output[1]=gender(:0), output[2]=ethnicity(:2)
            val genderOutput = Array(1) { FloatArray(1) }
            val ageOutput = Array(1) { FloatArray(1) }
            val ethnicityOutput = Array(1) { FloatArray(4) }

            val outputMap = HashMap<Int, Any>()
            outputMap[0] = ageOutput
            outputMap[1] = genderOutput
            outputMap[2] = ethnicityOutput

            LogCapture.d(TAG, "Running MoE inference...")
            interpreter!!.runForMultipleInputsOutputs(arrayOf(inputBuffer), outputMap)

            // Gender
            val genderProb = genderOutput[0][0]
            val gender = if (genderProb > 0.5f) Gender.FEMALE else Gender.MALE
            val genderConfidence = if (genderProb > 0.5f) genderProb else (1f - genderProb)

            // Age
            val age = ageOutput[0][0].toInt().coerceIn(0, 116)

            // Ethnicity
            val probs = ethnicityOutput[0]
            val maxIdx = probs.indices.maxByOrNull { probs[it] } ?: 0
            val ethnicity = ethnicityFromIndex(maxIdx)
            val ethnicityConfidence = probs[maxIdx]

            LogCapture.i(TAG, "MoE Result: Age=$age, Gender=${gender.label}, Ethnicity=${ethnicity.label}")

            PredictionResult(
                age = age,
                ageConfidence = 0.85f,
                gender = gender,
                genderConfidence = genderConfidence,
                ethnicity = ethnicity,
                ethnicityConfidence = ethnicityConfidence
            )
        } catch (e: Exception) {
            LogCapture.e(TAG, "MoE prediction failed: ${e.message}", e)
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
