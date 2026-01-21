package com.sae.facepredictor.ui.main

import android.Manifest
import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.widget.ScrollView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import com.sae.facepredictor.R
import com.sae.facepredictor.databinding.ActivityMainBinding
import com.sae.facepredictor.ui.auth.LoginActivity
import com.sae.facepredictor.ui.camera.CameraActivity
import com.sae.facepredictor.ui.history.HistoryActivity
import com.sae.facepredictor.ui.prediction.PredictionResultActivity
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.utils.LogCapture
import com.sae.facepredictor.utils.PredictionMode
import com.sae.facepredictor.utils.SessionManager
import com.sae.facepredictor.utils.showToast

class MainActivity : AppCompatActivity(), LogCapture.LogListener {

    companion object {
        private const val TAG = "MainActivity"
    }

    private lateinit var binding: ActivityMainBinding
    private lateinit var authService: FirebaseAuthService
    private lateinit var sessionManager: SessionManager

    private val galleryLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            navigateToPrediction(it.toString())
        }
    }

    private val cameraPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            openCamera()
        } else {
            showToast(getString(R.string.error_camera_permission))
        }
    }

    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            openGallery()
        } else {
            showToast(getString(R.string.error_storage_permission))
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        authService = FirebaseAuthService.getInstance()
        sessionManager = SessionManager(this)

        LogCapture.i(TAG, "MainActivity onCreate")

        setupUI()
        setupListeners()
        setupModelSwitch()
        setupLogs()
    }

    override fun onResume() {
        super.onResume()
        LogCapture.setLogListener(this)
        refreshLogs()
    }

    override fun onPause() {
        super.onPause()
        LogCapture.setLogListener(null)
    }

    override fun onNewLog(log: String) {
        runOnUiThread {
            val currentText = binding.tvLogs.text.toString()
            val newText = if (currentText == "Logs will appear here...") {
                log
            } else {
                "$currentText\n$log"
            }
            binding.tvLogs.text = newText
            // Auto-scroll to bottom
            binding.tvLogs.parent?.let { parent ->
                if (parent is ScrollView) {
                    parent.post { parent.fullScroll(ScrollView.FOCUS_DOWN) }
                }
            }
        }
    }

    private fun setupUI() {
        val username = authService.displayName ?: authService.userEmail?.substringBefore("@") ?: "Utilisateur"
        binding.tvWelcome.text = getString(R.string.welcome, username)

        binding.toolbar.setOnMenuItemClickListener { menuItem ->
            when (menuItem.itemId) {
                R.id.action_logout -> {
                    logout()
                    true
                }
                else -> false
            }
        }
        binding.toolbar.inflateMenu(R.menu.menu_main)
    }

    private fun setupListeners() {
        binding.cardCamera.setOnClickListener {
            checkCameraPermission()
        }

        binding.cardGallery.setOnClickListener {
            checkStoragePermission()
        }

        binding.cardHistory.setOnClickListener {
            startActivity(Intent(this, HistoryActivity::class.java))
        }

        binding.cardModelInfo.setOnClickListener {
            startActivity(Intent(this, ModelInfoActivity::class.java))
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                openCamera()
            }
            else -> {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun checkStoragePermission() {
        val permission = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            Manifest.permission.READ_MEDIA_IMAGES
        } else {
            Manifest.permission.READ_EXTERNAL_STORAGE
        }

        when {
            ContextCompat.checkSelfPermission(this, permission) == PackageManager.PERMISSION_GRANTED -> {
                openGallery()
            }
            else -> {
                storagePermissionLauncher.launch(permission)
            }
        }
    }

    private fun openCamera() {
        startActivity(Intent(this, CameraActivity::class.java))
    }

    private fun openGallery() {
        galleryLauncher.launch("image/*")
    }

    private fun navigateToPrediction(imagePath: String) {
        val intent = Intent(this, PredictionResultActivity::class.java).apply {
            putExtra(PredictionResultActivity.EXTRA_IMAGE_PATH, imagePath)
        }
        startActivity(intent)
    }

    private fun logout() {
        authService.logout()
        startActivity(Intent(this, LoginActivity::class.java))
        finish()
    }

    private fun setupModelSwitch() {
        // Initialize radio button state from preferences
        val currentMode = sessionManager.predictionMode
        when (currentMode) {
            PredictionMode.HYBRID -> binding.radioHybrid.isChecked = true
            PredictionMode.ORIENTED -> binding.radioOriented.isChecked = true
            PredictionMode.MULTITASK -> binding.radioMultitask.isChecked = true
        }

        // Handle radio button changes
        binding.radioGroupMode.setOnCheckedChangeListener { _, checkedId ->
            val newMode = when (checkedId) {
                R.id.radioHybrid -> PredictionMode.HYBRID
                R.id.radioOriented -> PredictionMode.ORIENTED
                R.id.radioMultitask -> PredictionMode.MULTITASK
                else -> PredictionMode.HYBRID
            }
            sessionManager.predictionMode = newMode
            LogCapture.i(TAG, "Model mode changed: ${newMode.label}")
            showToast("Mode ${newMode.label} activé")
        }
    }

    private fun setupLogs() {
        binding.btnCopyLogs.setOnClickListener {
            val logs = binding.tvLogs.text.toString()
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("FacePredictor Logs", logs)
            clipboard.setPrimaryClip(clip)
            showToast("Logs copiés!")
        }

        binding.btnRefreshLogs.setOnClickListener {
            refreshLogs()
        }

        binding.btnClearLogs.setOnClickListener {
            LogCapture.clear()
            binding.tvLogs.text = "Logs cleared."
            LogCapture.i(TAG, "Logs cleared by user")
        }

        // Initial log display
        refreshLogs()
    }

    private fun refreshLogs() {
        val appLogs = LogCapture.getLogsAsString()
        val logcatLogs = LogCapture.getRecentLogcat()

        val displayLogs = buildString {
            append("=== APP LOGS ===\n")
            if (appLogs.isNotEmpty()) {
                append(appLogs)
            } else {
                append("No app logs yet.")
            }
            append("\n\n=== LOGCAT (recent 100 lines) ===\n")
            append(logcatLogs)
        }

        binding.tvLogs.text = displayLogs

        // Scroll to bottom
        binding.tvLogs.parent?.let { parent ->
            if (parent is ScrollView) {
                parent.post { parent.fullScroll(ScrollView.FOCUS_DOWN) }
            }
        }
    }
}
