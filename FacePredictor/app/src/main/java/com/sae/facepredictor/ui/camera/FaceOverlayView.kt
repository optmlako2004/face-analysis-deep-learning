package com.sae.facepredictor.ui.camera

import android.content.Context
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
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
