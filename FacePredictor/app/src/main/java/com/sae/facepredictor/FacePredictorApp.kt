package com.sae.facepredictor

import android.app.Application
import androidx.appcompat.app.AppCompatDelegate
import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.FirebaseFirestoreSettings
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.utils.SessionManager

class FacePredictorApp : Application() {

    companion object {
        lateinit var instance: FacePredictorApp
            private set
    }

    override fun onCreate() {
        super.onCreate()
        instance = this

        try {
            FirebaseAuthService.getInstance()
        } catch (_: Exception) {}

        // Enable Firestore offline persistence (no crash without internet)
        try {
            val settings = FirebaseFirestoreSettings.Builder()
                .setPersistenceEnabled(true)
                .setCacheSizeBytes(FirebaseFirestoreSettings.CACHE_SIZE_UNLIMITED)
                .build()
            FirebaseFirestore.getInstance().firestoreSettings = settings
        } catch (_: Exception) {}

        // Apply dark mode preference at startup
        val sessionManager = SessionManager(this)
        AppCompatDelegate.setDefaultNightMode(
            when (sessionManager.darkMode) {
                1 -> AppCompatDelegate.MODE_NIGHT_YES
                0 -> AppCompatDelegate.MODE_NIGHT_NO
                else -> AppCompatDelegate.MODE_NIGHT_FOLLOW_SYSTEM
            }
        )
    }
}
