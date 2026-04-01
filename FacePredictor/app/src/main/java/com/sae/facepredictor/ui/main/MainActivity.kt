package com.sae.facepredictor.ui.main

import android.content.Intent
import android.content.res.Configuration
import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import androidx.navigation.fragment.NavHostFragment
import androidx.navigation.ui.setupWithNavController
import com.sae.facepredictor.R
import com.sae.facepredictor.databinding.ActivityMainNewBinding
import com.google.firebase.auth.FirebaseAuth
import com.sae.facepredictor.ui.tutorial.TutorialActivity
import com.sae.facepredictor.utils.LogCapture
import com.sae.facepredictor.utils.SessionManager

class MainActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainNewBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainNewBinding.inflate(layoutInflater)
        setContentView(binding.root)

        LogCapture.i(TAG, "MainActivity onCreate")

        setupNavigation()

        val sessionManager = SessionManager(this)
        val userId = FirebaseAuth.getInstance().currentUser?.uid
        if (userId != null && !sessionManager.hasSeenTutorial(userId)) {
            startActivity(Intent(this, TutorialActivity::class.java))
        }
    }

    override fun onConfigurationChanged(newConfig: Configuration) {
        super.onConfigurationChanged(newConfig)
        // Dark mode change handled without recreating the activity
        recreate()
    }

    private fun setupNavigation() {
        val navHostFragment = supportFragmentManager
            .findFragmentById(R.id.nav_host_fragment) as NavHostFragment
        val navController = navHostFragment.navController

        binding.bottomNavigation.setupWithNavController(navController)
    }
}
