package com.sae.facepredictor

import android.app.Application
import com.sae.facepredictor.data.firebase.FirebaseAuthService

class FacePredictorApp : Application() {

    companion object {
        lateinit var instance: FacePredictorApp
            private set
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
        FirebaseAuthService.init(this)
    }
}