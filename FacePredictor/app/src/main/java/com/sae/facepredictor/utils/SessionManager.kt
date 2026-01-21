package com.sae.facepredictor.utils

import android.content.Context
import android.content.SharedPreferences

class SessionManager(context: Context) {

    private val prefs: SharedPreferences =
        context.getSharedPreferences(PREF_NAME, Context.MODE_PRIVATE)

    companion object {
        private const val PREF_NAME = "face_predictor_session"
        private const val KEY_USER_ID = "user_id"
        private const val KEY_USERNAME = "username"
        private const val KEY_EMAIL = "email"
        private const val KEY_IS_LOGGED_IN = "is_logged_in"
        private const val KEY_USE_MULTITASK_MODEL = "use_multitask_model"
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
        prefs.edit().clear().apply()
    }

    val isLoggedIn: Boolean
        get() = prefs.getBoolean(KEY_IS_LOGGED_IN, false)

    val userId: Long
        get() = prefs.getLong(KEY_USER_ID, -1)

    val username: String?
        get() = prefs.getString(KEY_USERNAME, null)

    val email: String?
        get() = prefs.getString(KEY_EMAIL, null)

    // Model preference: true = Multitask V4, false = Oriented V2 (3 separate models)
    var useMultitaskModel: Boolean
        get() = prefs.getBoolean(KEY_USE_MULTITASK_MODEL, false) // Default: Oriented V2
        set(value) {
            prefs.edit().putBoolean(KEY_USE_MULTITASK_MODEL, value).apply()
        }
}
