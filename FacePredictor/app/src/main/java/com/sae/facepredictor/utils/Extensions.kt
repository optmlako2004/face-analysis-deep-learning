package com.sae.facepredictor.utils

import android.content.Context
import android.widget.Toast
import java.text.SimpleDateFormat
import java.util.*

fun Context.showToast(message: String, duration: Int = Toast.LENGTH_SHORT) {
    Toast.makeText(this, message, duration).show()
}

fun Long.toFormattedDate(): String {
    val sdf = SimpleDateFormat("dd/MM/yyyy HH:mm", Locale.getDefault())
    return sdf.format(Date(this))
}

fun Float.toPercentage(): String {
    return "%.1f%%".format(this * 100)
}

/**
 * Estimate age prediction confidence based on predicted age.
 * Age regression has no direct confidence output, so we estimate it:
 * - Ages 10-60 (most represented in UTKFace) → higher confidence
 * - Extreme ages (0-5, 80+) → lower confidence (fewer training samples)
 * - Based on model MAE of ~6 years
 */
fun estimateAgeConfidence(predictedAge: Int): Float {
    val age = predictedAge.coerceIn(0, 116)
    // Base confidence from MAE (~6 years on a 0-116 range)
    val baseConfidence = 0.88f
    // Penalty for extreme ages (fewer samples in dataset)
    val penalty = when {
        age in 15..55 -> 0f       // Well represented range
        age in 10..14 || age in 56..70 -> 0.05f
        age in 5..9 || age in 71..85 -> 0.12f
        else -> 0.20f             // Very young or very old
    }
    return (baseConfidence - penalty).coerceIn(0.55f, 0.95f)
}
