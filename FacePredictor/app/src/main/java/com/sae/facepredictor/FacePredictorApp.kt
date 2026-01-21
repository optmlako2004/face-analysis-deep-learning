package com.sae.facepredictor

import android.app.Application
import com.sae.facepredictor.data.database.AppDatabase

class FacePredictorApp : Application() {

    val database: AppDatabase by lazy {
        AppDatabase.getDatabase(this)
    }

    companion object {
        lateinit var instance: FacePredictorApp
            private set
    }

    override fun onCreate() {
        super.onCreate()
        instance = this
    }
}
