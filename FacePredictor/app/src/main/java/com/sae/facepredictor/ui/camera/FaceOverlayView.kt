package com.sae.facepredictor.ui.camera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View

class FaceOverlayView @JvmOverloads constructor(
    context: Context,
    attrs: AttributeSet? = null,
    defStyleAttr: Int = 0
) : View(context, attrs, defStyleAttr) {

    private val facePaint = Paint().apply {
        color = Color.GREEN
        style = Paint.Style.STROKE
        strokeWidth = 4f
    }

    private val guidePaint = Paint().apply {
        color = Color.WHITE
        style = Paint.Style.STROKE
        strokeWidth = 3f
        alpha = 120
    }

    private val dimPaint = Paint().apply {
        color = Color.BLACK
        alpha = 80
    }

    private val faces = mutableListOf<RectF>()
    private var imageWidth: Int = 1
    private var imageHeight: Int = 1
    private var isFrontCamera: Boolean = true

    fun setFaces(
        detectedFaces: List<RectF>,
        imgWidth: Int,
        imgHeight: Int,
        frontCamera: Boolean
    ) {
        faces.clear()
        faces.addAll(detectedFaces)
        imageWidth = imgWidth
        imageHeight = imgHeight
        isFrontCamera = frontCamera
        invalidate()
    }

    fun clearFaces() {
        faces.clear()
        invalidate()
    }

    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)

        if (faces.isEmpty()) {
            drawGuideOval(canvas)
        } else {
            drawDetectedFaces(canvas)
        }
    }

    private fun drawGuideOval(canvas: Canvas) {
        val centerX = width / 2f
        val centerY = height * 0.38f
        val ovalWidth = width * 0.55f
        val ovalHeight = ovalWidth * 1.35f

        val ovalRect = RectF(
            centerX - ovalWidth / 2,
            centerY - ovalHeight / 2,
            centerX + ovalWidth / 2,
            centerY + ovalHeight / 2
        )

        // Dim area outside the oval
        val path = Path()
        path.addRect(0f, 0f, width.toFloat(), height.toFloat(), Path.Direction.CW)
        path.addOval(ovalRect, Path.Direction.CCW)
        canvas.drawPath(path, dimPaint)

        // Draw oval border
        canvas.drawOval(ovalRect, guidePaint)
    }

    private fun drawDetectedFaces(canvas: Canvas) {
        val scaleX = width.toFloat() / imageWidth
        val scaleY = height.toFloat() / imageHeight

        for (face in faces) {
            val left = if (isFrontCamera) {
                width - (face.right * scaleX)
            } else {
                face.left * scaleX
            }
            val right = if (isFrontCamera) {
                width - (face.left * scaleX)
            } else {
                face.right * scaleX
            }

            val scaledRect = RectF(
                left,
                face.top * scaleY,
                right,
                face.bottom * scaleY
            )

            canvas.drawRect(scaledRect, facePaint)
        }
    }
}
