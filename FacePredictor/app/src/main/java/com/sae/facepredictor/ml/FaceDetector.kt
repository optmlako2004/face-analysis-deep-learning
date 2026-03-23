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

    /**
     * Crop a square region centered on the face.
     * Uses max(width, height) of the bbox as the base side length,
     * then applies padding. Shifts the crop before clipping to keep it square.
     * Designed for MoE model to match UTKFace training distribution (~80% face).
     */
    fun cropFaceSquare(bitmap: Bitmap, faceRect: RectF, padding: Float = 0.15f): Bitmap {
        val imgW = bitmap.width
        val imgH = bitmap.height

        // Use the larger side of the bbox as base
        val bboxW = faceRect.width()
        val bboxH = faceRect.height()
        val baseSide = maxOf(bboxW, bboxH)

        // Apply padding to get the full crop side
        val cropSide = (baseSide * (1f + 2f * padding)).toInt()

        // Center of the face bbox
        val cx = ((faceRect.left + faceRect.right) / 2f).toInt()
        val cy = ((faceRect.top + faceRect.bottom) / 2f).toInt()

        // Initial top-left corner
        var left = cx - cropSide / 2
        var top = cy - cropSide / 2

        // Shift to stay within image bounds (keep square before clipping)
        if (left < 0) left = 0
        if (top < 0) top = 0
        if (left + cropSide > imgW) left = maxOf(0, imgW - cropSide)
        if (top + cropSide > imgH) top = maxOf(0, imgH - cropSide)

        // Final clipped dimensions (may be smaller than cropSide if image is tiny)
        val finalW = minOf(cropSide, imgW - left)
        val finalH = minOf(cropSide, imgH - top)
        val finalSide = minOf(finalW, finalH)

        LogCapture.d(TAG, "cropFaceSquare: bbox=${bboxW.toInt()}x${bboxH.toInt()}, cropSide=$cropSide, final=${finalSide}x${finalSide}, at ($left,$top)")

        return if (finalSide > 0) {
            Bitmap.createBitmap(bitmap, left, top, finalSide, finalSide)
        } else {
            bitmap
        }
    }

    fun close() {
        faceDetector?.close()
        faceDetector = null
    }
}
