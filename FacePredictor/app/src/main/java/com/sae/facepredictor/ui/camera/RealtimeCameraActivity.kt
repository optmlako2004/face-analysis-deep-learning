package com.sae.facepredictor.ui.camera

import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Matrix
import android.graphics.RectF
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.sae.facepredictor.R
import com.sae.facepredictor.data.model.Gender
import com.sae.facepredictor.data.model.PredictionResult
import com.sae.facepredictor.databinding.ActivityRealtimeCameraBinding
import com.sae.facepredictor.ml.FaceDetectorHelper
import com.sae.facepredictor.ml.FacePredictorHybrid
import com.sae.facepredictor.ml.FacePredictorMoE
import com.sae.facepredictor.ml.FacePredictorModel
import com.sae.facepredictor.ml.FacePredictorModelV2
import com.sae.facepredictor.ui.prediction.PredictionResultActivity
import com.sae.facepredictor.utils.LogCapture
import com.sae.facepredictor.utils.PredictionMode
import com.sae.facepredictor.utils.SessionManager
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicBoolean

class RealtimeCameraActivity : AppCompatActivity() {

    companion object {
        private const val TAG = "RealtimeCameraActivity"
        private const val ANALYSIS_INTERVAL_MS = 500L
        const val EXTRA_REALTIME_RESULT = "extra_realtime_result"
        const val EXTRA_AGE = "extra_age"
        const val EXTRA_AGE_CONFIDENCE = "extra_age_confidence"
        const val EXTRA_GENDER = "extra_gender"
        const val EXTRA_GENDER_CONFIDENCE = "extra_gender_confidence"
        const val EXTRA_ETHNICITY = "extra_ethnicity"
        const val EXTRA_ETHNICITY_CONFIDENCE = "extra_ethnicity_confidence"
    }

    private lateinit var binding: ActivityRealtimeCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private lateinit var sessionManager: SessionManager

    private var cameraProvider: ProcessCameraProvider? = null
    private var imageCapture: ImageCapture? = null
    private var isFrontCamera = true

    // Face detection & prediction
    private var faceDetector: FaceDetectorHelper? = null
    private var predictorHybrid: FacePredictorHybrid? = null
    private var predictorV2: FacePredictorModelV2? = null
    private var predictorV4: FacePredictorModel? = null
    private var predictorMoE: FacePredictorMoE? = null

    // State
    private val isProcessing = AtomicBoolean(false)
    private var lastAnalysisTime = 0L
    private var currentResult: PredictionResult? = null
    private var currentBitmap: Bitmap? = null
    private var currentFaceRect: RectF? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRealtimeCameraBinding.inflate(layoutInflater)
        setContentView(binding.root)

        sessionManager = SessionManager(this)
        cameraExecutor = Executors.newSingleThreadExecutor()

        LogCapture.i(TAG, "RealtimeCameraActivity onCreate - Mode: ${sessionManager.predictionMode}")

        initializeModels()
        setupListeners()
        startCamera()
    }

    private fun initializeModels() {
        lifecycleScope.launch(Dispatchers.IO) {
            try {
                LogCapture.d(TAG, "Initializing models...")
                faceDetector = FaceDetectorHelper(this@RealtimeCameraActivity)

                when (sessionManager.predictionMode) {
                    PredictionMode.HYBRID -> {
                        predictorHybrid = FacePredictorHybrid(this@RealtimeCameraActivity)
                        LogCapture.i(TAG, "Hybrid predictor initialized")
                    }
                    PredictionMode.ORIENTED -> {
                        predictorV2 = FacePredictorModelV2(this@RealtimeCameraActivity)
                        LogCapture.i(TAG, "V2 EfficientNet predictor initialized")
                    }
                    PredictionMode.MULTITASK -> {
                        predictorV4 = FacePredictorModel(this@RealtimeCameraActivity)
                        LogCapture.i(TAG, "V4 Multitask EfficientNet predictor initialized")
                    }
                    PredictionMode.MOE -> {
                        predictorMoE = FacePredictorMoE(this@RealtimeCameraActivity)
                        LogCapture.i(TAG, "MoE predictor initialized")
                    }
                }

                withContext(Dispatchers.Main) {
                    binding.tvStatus.text = getString(R.string.realtime_ready)
                }
            } catch (e: Exception) {
                LogCapture.e(TAG, "Failed to initialize models: ${e.message}", e)
            }
        }
    }

    private fun setupListeners() {
        binding.btnClose.setOnClickListener {
            finish()
        }

        binding.btnSwitchCamera.setOnClickListener {
            isFrontCamera = !isFrontCamera
            startCamera()
        }

        binding.btnCapture.setOnClickListener {
            captureAndSave()
        }
    }

    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)

        cameraProviderFuture.addListener({
            cameraProvider = cameraProviderFuture.get()

            // Preview
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.previewView.surfaceProvider)
                }

            // Image capture for saving
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
                .build()

            // Image analysis for real-time processing
            val imageAnalyzer = ImageAnalysis.Builder()
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { imageProxy ->
                        processImage(imageProxy)
                    }
                }

            val cameraSelector = if (isFrontCamera) {
                CameraSelector.DEFAULT_FRONT_CAMERA
            } else {
                CameraSelector.DEFAULT_BACK_CAMERA
            }

            try {
                cameraProvider?.unbindAll()
                cameraProvider?.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageCapture,
                    imageAnalyzer
                )
                LogCapture.i(TAG, "Camera bound successfully (front: $isFrontCamera)")
            } catch (e: Exception) {
                LogCapture.e(TAG, "Camera binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }

    private fun processImage(imageProxy: ImageProxy) {
        val currentTime = System.currentTimeMillis()

        // Throttle analysis
        if (currentTime - lastAnalysisTime < ANALYSIS_INTERVAL_MS) {
            imageProxy.close()
            return
        }

        // Skip if already processing
        if (!isProcessing.compareAndSet(false, true)) {
            imageProxy.close()
            return
        }

        lastAnalysisTime = currentTime

        try {
            val bitmap = imageProxyToBitmap(imageProxy)
            imageProxy.close()

            if (bitmap == null) {
                isProcessing.set(false)
                return
            }

            // Store for potential capture
            currentBitmap = bitmap

            // Detect faces
            val detectionResult = faceDetector?.detectFaces(bitmap)

            if (detectionResult == null || detectionResult.faces.isEmpty()) {
                currentFaceRect = null
                currentResult = null
                runOnUiThread {
                    showNoFaceDetected()
                    binding.faceOverlay.clearFaces()
                }
                isProcessing.set(false)
                return
            }

            // Get the largest face
            val largestFace = detectionResult.faces.maxByOrNull {
                (it.right - it.left) * (it.bottom - it.top)
            } ?: detectionResult.faces.first()

            currentFaceRect = largestFace

            // Update face overlay
            runOnUiThread {
                binding.faceOverlay.setFaces(
                    listOf(largestFace),
                    bitmap.width,
                    bitmap.height,
                    isFrontCamera
                )
            }

            // Crop face and predict
            val croppedFace = faceDetector?.cropFace(bitmap, largestFace, 0.5f) ?: bitmap
            val result = predict(croppedFace)

            if (result != null) {
                currentResult = result
                runOnUiThread {
                    showResults(result)
                }
            }

        } catch (e: Exception) {
            LogCapture.e(TAG, "Error processing image: ${e.message}", e)
        } finally {
            isProcessing.set(false)
        }
    }

    private fun predict(bitmap: Bitmap): PredictionResult? {
        return when (sessionManager.predictionMode) {
            PredictionMode.HYBRID -> predictorHybrid?.predict(bitmap)
            PredictionMode.ORIENTED -> predictorV2?.predict(bitmap)
            PredictionMode.MULTITASK -> predictorV4?.predict(bitmap)
            PredictionMode.MOE -> predictorMoE?.predict(bitmap)
        }
    }

    private fun imageProxyToBitmap(imageProxy: ImageProxy): Bitmap? {
        return try {
            var bitmap = imageProxy.toBitmap()

            val rotation = imageProxy.imageInfo.rotationDegrees
            if (rotation != 0 || isFrontCamera) {
                val matrix = Matrix()
                matrix.postRotate(rotation.toFloat())
                if (isFrontCamera) {
                    matrix.postScale(-1f, 1f, bitmap.width / 2f, bitmap.height / 2f)
                }
                bitmap = Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
            }

            bitmap
        } catch (e: Exception) {
            LogCapture.e(TAG, "Failed to convert ImageProxy to Bitmap", e)
            null
        }
    }

    private fun showNoFaceDetected() {
        binding.tvStatus.text = getString(R.string.realtime_searching)
        binding.statusIndicator.setBackgroundResource(R.drawable.circle_status_inactive)
        binding.tvNoFace.visibility = View.VISIBLE
        binding.resultsContainer.visibility = View.GONE
    }

    private fun showResults(result: PredictionResult) {
        binding.tvStatus.text = getString(R.string.realtime_analyzing)
        binding.statusIndicator.setBackgroundResource(R.drawable.circle_status_active)
        binding.tvNoFace.visibility = View.GONE
        binding.resultsContainer.visibility = View.VISIBLE

        // Age
        binding.tvAge.text = "${result.age}"
        binding.tvAgeConfidence.text = "${(result.ageConfidence * 100).toInt()}%"

        // Gender
        binding.tvGender.text = result.gender.label
        val genderColor = if (result.gender == Gender.MALE) {
            ContextCompat.getColor(this, R.color.male_color)
        } else {
            ContextCompat.getColor(this, R.color.female_color)
        }
        binding.tvGender.setTextColor(genderColor)
        binding.tvGenderConfidence.text = "${(result.genderConfidence * 100).toInt()}%"

        // Ethnicity
        binding.tvEthnicity.text = result.ethnicity.label
        binding.tvEthnicityConfidence.text = "${(result.ethnicityConfidence * 100).toInt()}%"
    }

    private fun captureAndSave() {
        val bitmap = currentBitmap ?: return
        val result = currentResult ?: return

        setLoading(true)
        LogCapture.i(TAG, "Capturing frame with result: $result")

        lifecycleScope.launch(Dispatchers.IO) {
            try {
                // Save bitmap to file
                val file = saveBitmapToFile(bitmap)

                withContext(Dispatchers.Main) {
                    if (file != null) {
                        // Navigate to PredictionResultActivity with pre-computed result
                        val intent = Intent(this@RealtimeCameraActivity, PredictionResultActivity::class.java).apply {
                            putExtra(PredictionResultActivity.EXTRA_IMAGE_PATH, file.absolutePath)
                            putExtra(EXTRA_REALTIME_RESULT, true)
                            putExtra(EXTRA_AGE, result.age)
                            putExtra(EXTRA_AGE_CONFIDENCE, result.ageConfidence)
                            putExtra(EXTRA_GENDER, result.gender.name)
                            putExtra(EXTRA_GENDER_CONFIDENCE, result.genderConfidence)
                            putExtra(EXTRA_ETHNICITY, result.ethnicity.name)
                            putExtra(EXTRA_ETHNICITY_CONFIDENCE, result.ethnicityConfidence)
                        }
                        startActivity(intent)
                        finish()
                    } else {
                        LogCapture.e(TAG, "Failed to save bitmap")
                        setLoading(false)
                    }
                }
            } catch (e: Exception) {
                LogCapture.e(TAG, "Capture failed: ${e.message}", e)
                withContext(Dispatchers.Main) {
                    setLoading(false)
                }
            }
        }
    }

    private fun saveBitmapToFile(bitmap: Bitmap): File? {
        return try {
            val file = File(cacheDir, "realtime_${System.currentTimeMillis()}.jpg")
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            }
            file
        } catch (e: Exception) {
            LogCapture.e(TAG, "Failed to save bitmap", e)
            null
        }
    }

    private fun setLoading(loading: Boolean) {
        binding.loadingOverlay.visibility = if (loading) View.VISIBLE else View.GONE
        binding.btnCapture.isEnabled = !loading
    }

    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
        faceDetector?.close()
        predictorHybrid?.close()
        predictorV2?.close()
        predictorV4?.close()
        predictorMoE?.close()
        LogCapture.d(TAG, "RealtimeCameraActivity destroyed")
    }
}
