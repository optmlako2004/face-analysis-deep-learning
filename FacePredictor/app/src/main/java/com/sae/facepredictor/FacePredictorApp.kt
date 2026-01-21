package com.sae.facepredictor

import android.app.Application

class FacePredictorApp : Application() {

    companion object {
        lateinit var instance: FacePredictorApp
            private set
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }
}
