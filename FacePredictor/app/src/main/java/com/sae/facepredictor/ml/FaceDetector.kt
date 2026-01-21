package com.sae.facepredictor.ml

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import com.google.mediapipe.framework.image.BitmapImageBuilder
import com.google.mediapipe.tasks.core.BaseOptions
import com.google.mediapipe.tasks.vision.facedetector.FaceDetector
import com.google.mediapipe.tasks.vision.facedetector.FaceDetectorResult
import com.sae.facepredictor.utils.LogCapture

class FaceDetectorHelper(private val context: Context) {

    companion object {
        private const val TAG = "FaceDetectorHelper"
    }

    private var faceDetector: FaceDetector? = null

    data class FaceDetectionResult(
        val faces: List<RectF>,
        val success: Boolean,
        val errorMessage: String? = null
    )

    init {
        setupFaceDetector()
    }

    private fun setupFaceDetector() {
        try {
            LogCapture.d(TAG, "Setting up face detector...")
            val baseOptions = BaseOptions.builder()
                .setModelAssetPath("face_detection_short_range.tflite")
                .build()

            val options = FaceDetector.FaceDetectorOptions.builder()
                .setBaseOptions(baseOptions)
                .setMinDetectionConfidence(0.5f)
                .setRunningMode(com.google.mediapipe.tasks.vision.core.RunningMode.IMAGE)
                .build()

            faceDetector = FaceDetector.createFromOptions(context, options)
            LogCapture.i(TAG, "Face detector initialized successfully")
        } catch (e: Exception) {
            LogCapture.e(TAG, "Failed to setup face detector: ${e.message}", e)
        }
    }

    fun detectFaces(bitmap: Bitmap): FaceDetectionResult {
        LogCapture.d(TAG, "Detecting faces in bitmap ${bitmap.width}x${bitmap.height}")

        if (faceDetector == null) {
            LogCapture.e(TAG, "Face detector not initialized")
            return FaceDetectionResult(
                faces = emptyList(),
                success = false,
                errorMessage = "Face detector not initialized"
            )
        }

        return try {
            val mpImage = BitmapImageBuilder(bitmap).build()
            val result: FaceDetectorResult = faceDetector!!.detect(mpImage)

            val faces = result.detections().map { detection ->
                val bbox = detection.boundingBox()
                RectF(
                    bbox.left,
                    bbox.top,
                    bbox.right,
                    bbox.bottom
                )
            }

            LogCapture.d(TAG, "Detected ${faces.size} face(s)")
            FaceDetectionResult(
                faces = faces,
                success = true
            )
        } catch (e: Exception) {
            LogCapture.e(TAG, "Face detection failed: ${e.message}", e)
            FaceDetectionResult(
                faces = emptyList(),
                success = false,
                errorMessage = e.message
            )
        }
    }

    fun cropFace(bitmap: Bitmap, faceRect: RectF, padding: Float = 0.2f): Bitmap {
        val width = bitmap.width
        val height = bitmap.height

        // Add padding around face
        val padX = (faceRect.width() * padding).toInt()
        val padY = (faceRect.height() * padding).toInt()

        val left = maxOf(0, (faceRect.left - padX).toInt())
        val top = maxOf(0, (faceRect.top - padY).toInt())
        val right = minOf(width, (faceRect.right + padX).toInt())
        val bottom = minOf(height, (faceRect.bottom + padY).toInt())

        val cropWidth = right - left
        val cropHeight = bottom - top

        return if (cropWidth > 0 && cropHeight > 0) {
            Bitmap.createBitmap(bitmap, left, top, cropWidth, cropHeight)
        } else {
            bitmap
        }
    }

    fun close() {
        faceDetector?.close()
        faceDetector = null
    }
}
