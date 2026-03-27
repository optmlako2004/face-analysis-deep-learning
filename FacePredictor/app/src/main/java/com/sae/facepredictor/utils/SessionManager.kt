package com.sae.facepredictor.utils

import android.content.Context
import android.content.SharedPreferences

enum class PredictionMode(val value: Int, val label: String, val description: String) {
    ORIENTED(0, "Orienté V2", "3 modèles spécialisés EfficientNet"),
    MULTITASK(1, "Multitâche V4", "1 modèle unifié EfficientNet"),
    HYBRID(2, "Hybride", "Combine le meilleur des deux modèles"),
    MOE(4, "MoE Expert", "Mixture of Experts - MobileNetV3");

    companion object {
        fun fromValue(value: Int): PredictionMode {
            return entries.find { it.value == value } ?: MOE
        }
    }
}

class SessionManager(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val PREF_NAME = "face_predictor_session"
        private const val KEY_USER_ID = "user_id"
        private const val KEY_USERNAME = "username"
        private const val KEY_EMAIL = "email"
        private const val KEY_IS_LOGGED_IN = "is_logged_in"
        private const val KEY_PREDICTION_MODE = "prediction_mode"
        private const val KEY_HAS_SEEN_TUTORIAL = "has_seen_tutorial"
        private const val KEY_DARK_MODE = "dark_mode"
    }

    fun saveSession(userId: Long, username: String, email: String) {
        prefs.edit().apply {
            putLong(KEY_USER_ID, userId)
            putString(KEY_USERNAME, username)
            putString(KEY_EMAIL, email)
            putBoolean(KEY_IS_LOGGED_IN, true)
            apply()
        }
    }

    fun clearSession() {
        prefs.edit().apply {
            remove(KEY_USER_ID)
            remove(KEY_USERNAME)
            remove(KEY_EMAIL)
            remove(KEY_IS_LOGGED_IN)
            remove(KEY_PREDICTION_MODE)
            apply()
        }
    }

    val isLoggedIn: Boolean
        get() = prefs.getBoolean(KEY_IS_LOGGED_IN, false)

    val userId: Long
        get() = prefs.getLong(KEY_USER_ID, -1)

    val username: String?
        get() = prefs.getString(KEY_USERNAME, null)

    val email: String?
        get() = prefs.getString(KEY_EMAIL, null)

    // Prediction mode: ORIENTED, MULTITASK, or HYBRID (default)
    var predictionMode: PredictionMode
        get() = PredictionMode.fromValue(prefs.getInt(KEY_PREDICTION_MODE, PredictionMode.MOE.value))
        set(value) {
            prefs.edit().putInt(KEY_PREDICTION_MODE, value.value).apply()
        }

    fun hasSeenTutorial(userId: String): Boolean =
        prefs.getBoolean("${KEY_HAS_SEEN_TUTORIAL}_$userId", false)

    fun setHasSeenTutorial(userId: String, value: Boolean) =
        prefs.edit().putBoolean("${KEY_HAS_SEEN_TUTORIAL}_$userId", value).apply()

    // Dark mode: -1 = system default, 0 = light, 1 = dark
    var darkMode: Int
        get() = prefs.getInt(KEY_DARK_MODE, -1)
        set(value) {
            prefs.edit().putInt(KEY_DARK_MODE, value).apply()
        }

    // Backward compatibility
    var useMultitaskModel: Boolean
        get() = predictionMode == PredictionMode.MULTITASK
        set(value) {
            predictionMode = if (value) PredictionMode.MULTITASK else PredictionMode.ORIENTED
        }
}
