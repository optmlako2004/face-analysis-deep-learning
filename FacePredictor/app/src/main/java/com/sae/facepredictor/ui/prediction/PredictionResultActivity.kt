package com.sae.facepredictor.ui.prediction

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.exifinterface.media.ExifInterface
import androidx.lifecycle.lifecycleScope
import coil.load
import com.sae.facepredictor.R
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.data.firebase.FirestoreRepository
import com.sae.facepredictor.data.model.Ethnicity
import com.sae.facepredictor.data.model.Gender
import com.sae.facepredictor.data.model.PredictionResult
import com.sae.facepredictor.ui.camera.RealtimeCameraActivity
import com.sae.facepredictor.databinding.ActivityPredictionResultBinding
import com.sae.facepredictor.ml.FaceDetectorHelper
import com.sae.facepredictor.ml.FacePredictorHybrid

import com.sae.facepredictor.ml.FacePredictorModel
import com.sae.facepredictor.ml.FacePredictorModelV2
import com.sae.facepredictor.utils.LogCapture
import com.sae.facepredictor.utils.PredictionMode
import com.sae.facepredictor.utils.SessionManager
import com.sae.facepredictor.utils.showToast
import com.sae.facepredictor.utils.toPercentage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class PredictionResultActivity : AppCompatActivity() {

    private lateinit var binding: ActivityPredictionResultBinding
    private lateinit var authService: FirebaseAuthService
    private lateinit var firestoreRepository: FirestoreRepository
    private lateinit var sessionManager: SessionManager

    // Support for all model types
    private var predictorV2: FacePredictorModelV2? = null
    private var predictorV4: FacePredictorModel? = null
    private var predictorHybrid: FacePredictorHybrid? = null
    private var predictionMode: PredictionMode = PredictionMode.HYBRID
    private var faceDetector: FaceDetectorHelper? = null

    private var imagePath: String? = null
    private var currentResult: PredictionResult? = null
    private var savedImagePath: String? = null

    companion object {
        const val EXTRA_IMAGE_PATH = "extra_image_path"
        private const val TAG = "PredictionResult"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityPredictionResultBinding.inflate(layoutInflater)
        setContentView(binding.root)

        authService = FirebaseAuthService.getInstance()
        firestoreRepository = FirestoreRepository.getInstance()
        sessionManager = SessionManager(this)
        predictionMode = sessionManager.predictionMode

        imagePath = intent.getStringExtra(EXTRA_IMAGE_PATH)
        LogCapture.d(TAG, "Received image path: $imagePath")
        LogCapture.d(TAG, "Using model: ${predictionMode.label}")

        setupUI()

        // Check if this is a realtime result (pre-computed)
        val isRealtimeResult = intent.getBooleanExtra(RealtimeCameraActivity.EXTRA_REALTIME_RESULT, false)

        if (isRealtimeResult) {
            // Use pre-computed result from realtime camera
            handleRealtimeResult()
        } else {
            // Initialize predictor and process image
            lifecycleScope.launch {
                initAndProcess()
            }
        }
    }

    private fun handleRealtimeResult() {
        LogCapture.d(TAG, "Handling pre-computed realtime result")

        // Extract result from intent
        val age = intent.getIntExtra(RealtimeCameraActivity.EXTRA_AGE, 0)
        val ageConfidence = intent.getFloatExtra(RealtimeCameraActivity.EXTRA_AGE_CONFIDENCE, 0f)
        val genderName = intent.getStringExtra(RealtimeCameraActivity.EXTRA_GENDER) ?: Gender.MALE.name
        val genderConfidence = intent.getFloatExtra(RealtimeCameraActivity.EXTRA_GENDER_CONFIDENCE, 0f)
        val ethnicityName = intent.getStringExtra(RealtimeCameraActivity.EXTRA_ETHNICITY) ?: Ethnicity.WHITE.name
        val ethnicityConfidence = intent.getFloatExtra(RealtimeCameraActivity.EXTRA_ETHNICITY_CONFIDENCE, 0f)

        val result = PredictionResult(
            age = age,
            ageConfidence = ageConfidence,
            gender = Gender.valueOf(genderName),
            genderConfidence = genderConfidence,
            ethnicity = Ethnicity.valueOf(ethnicityName),
            ethnicityConfidence = ethnicityConfidence
        )

        // Store for saving
        currentResult = result
        savedImagePath = imagePath

        // Load and display image
        imagePath?.let { path ->
            binding.ivPreview.load(File(path))
        }

        // Display result directly
        displayResult(result)
    }

    private fun setupUI() {
        binding.toolbar.setNavigationOnClickListener {
            finish()
        }

        binding.btnSave.setOnClickListener {
            savePrediction()
        }

        binding.btnSave.isEnabled = false
    }

    private suspend fun initAndProcess() {
        setLoading(true)

        try {
            // Initialize the appropriate predictor based on user preference
            withContext(Dispatchers.IO) {
                when (predictionMode) {
                    PredictionMode.HYBRID -> {
                        predictorHybrid = FacePredictorHybrid(this@PredictionResultActivity)
                        LogCapture.d(TAG, "Predictor HYBRID initialized (EfficientNet)")
                    }
                    PredictionMode.ORIENTED -> {
                        predictorV2 = FacePredictorModelV2(this@PredictionResultActivity)
                        LogCapture.d(TAG, "Predictor V2 initialized (3 separate EfficientNet models)")
                    }
                    PredictionMode.MULTITASK -> {
                        predictorV4 = FacePredictorModel(this@PredictionResultActivity)
                        LogCapture.d(TAG, "Predictor V4 Multitask initialized (1 unified EfficientNet model)")
                    }
                    // MoE removed - using only EfficientNetB0 models
                }
            }

            // Initialize face detector separately (can fail without blocking)
            try {
                withContext(Dispatchers.Main) {
                    faceDetector = FaceDetectorHelper(this@PredictionResultActivity)
                }
                LogCapture.d(TAG, "FaceDetector initialized")
            } catch (e: Exception) {
                LogCapture.e(TAG, "FaceDetector init failed (continuing without): ${e.message}")
                faceDetector = null
            }

            processImage()
        } catch (e: Exception) {
            LogCapture.e(TAG, "Failed to initialize: ${e.message}", e)
            showError("Erreur d'initialisation: ${e.message}")
        }
    }

    private suspend fun processImage() {
        val path = imagePath
        if (path == null) {
            showError("Chemin d'image manquant")
            return
        }

        try {
            // Load bitmap with EXIF rotation correction
            val bitmap = withContext(Dispatchers.IO) {
                loadBitmap(path)
            }

            if (bitmap == null) {
                showError("Impossible de charger l'image")
                return
            }

            LogCapture.d(TAG, "Bitmap loaded: ${bitmap.width}x${bitmap.height}")

            // Display original image
            withContext(Dispatchers.Main) {
                binding.ivPreview.load(bitmap)
            }

            // Try to detect face
            val faceCropResult = try {
                withContext(Dispatchers.Default) {
                    detectAndCropFace(bitmap)
                }
            } catch (e: Exception) {
                LogCapture.e(TAG, "Face detection error: ${e.message}")
                FaceCropResult(bitmap, true, 1) // On error, try with original
            }

            // Check if face was detected
            if (!faceCropResult.faceDetected || faceCropResult.bitmap == null) {
                LogCapture.d(TAG, "No face detected in image")
                showNoFaceDetected()
                return
            }

            val faceBitmap = faceCropResult.bitmap
            LogCapture.d(TAG, "Image for prediction: ${faceBitmap.width}x${faceBitmap.height}")

            // Save image for history
            savedImagePath = withContext(Dispatchers.IO) {
                saveBitmapToCache(faceBitmap)
            }

            // Run prediction using the appropriate model
            val result = withContext(Dispatchers.Default) {
                try {
                    when (predictionMode) {
                        PredictionMode.HYBRID -> predictorHybrid?.predict(faceBitmap)
                        PredictionMode.ORIENTED -> predictorV2?.predict(faceBitmap)
                        PredictionMode.MULTITASK -> predictorV4?.predict(faceBitmap)
                    }
                } catch (e: Exception) {
                    LogCapture.e(TAG, "Prediction exception: ${e.message}", e)
                    null
                }
            }

            if (result == null) {
                showError("La prédiction a échoué")
                return
            }

            LogCapture.d(TAG, "Prediction successful: age=${result.age}, gender=${result.gender}, ethnicity=${result.ethnicity}")

            currentResult = result
            displayResult(result)

        } catch (e: Exception) {
            LogCapture.e(TAG, "Processing error: ${e.message}", e)
            showError("Erreur: ${e.message}")
        }
    }

    data class FaceCropResult(
        val bitmap: Bitmap?,
        val faceDetected: Boolean,
        val faceCount: Int
    )

    private fun detectAndCropFace(bitmap: Bitmap): FaceCropResult {
        val detector = faceDetector ?: return FaceCropResult(bitmap, true, 1) // If no detector, assume face exists

        return try {
            val result = detector.detectFaces(bitmap)
            LogCapture.d(TAG, "Face detection: success=${result.success}, faces=${result.faces.size}")

            if (result.success && result.faces.isNotEmpty()) {
                val faceRect = result.faces.first()
                val croppedBitmap = detector.cropFace(bitmap, faceRect, padding = 0.3f)
                FaceCropResult(croppedBitmap, true, result.faces.size)
            } else {
                FaceCropResult(null, false, 0)
            }
        } catch (e: Exception) {
            LogCapture.e(TAG, "Face detection failed: ${e.message}")
            FaceCropResult(bitmap, true, 1) // On error, try with original
        }
    }

    private fun loadBitmap(path: String): Bitmap? {
        return try {
            if (path.startsWith("content://")) {
                val uri = Uri.parse(path)
                // Load bitmap
                val bitmap = contentResolver.openInputStream(uri)?.use { stream ->
                    BitmapFactory.decodeStream(stream)
                }
                
                if (bitmap == null) {
                    LogCapture.e(TAG, "Failed to decode bitmap from URI")
                    return null
                }
                
                // Get rotation from EXIF and apply it
                val rotation = getRotationFromUri(uri)
                LogCapture.d(TAG, "EXIF rotation: $rotation degrees")
                
                if (rotation != 0) {
                    rotateBitmap(bitmap, rotation)
                } else {
                    bitmap
                }
            } else {
                // File path
                val bitmap = BitmapFactory.decodeFile(path)
                if (bitmap == null) {
                    LogCapture.e(TAG, "Failed to decode bitmap from file")
                    return null
                }
                
                // Get rotation from file EXIF
                val exif = ExifInterface(path)
                val rotation = getRotationFromExif(exif)
                LogCapture.d(TAG, "EXIF rotation: $rotation degrees")
                
                if (rotation != 0) {
                    rotateBitmap(bitmap, rotation)
                } else {
                    bitmap
                }
            }
        } catch (e: Exception) {
            LogCapture.e(TAG, "Failed to load bitmap: ${e.message}", e)
            null
        }
    }

    private fun getRotationFromUri(uri: Uri): Int {
        return try {
            contentResolver.openInputStream(uri)?.use { inputStream ->
                val exif = ExifInterface(inputStream)
                getRotationFromExif(exif)
            } ?: 0
        } catch (e: Exception) {
            LogCapture.e(TAG, "Failed to get EXIF from URI: ${e.message}")
            0
        }
    }

    private fun getRotationFromExif(exif: ExifInterface): Int {
        val orientation = exif.getAttributeInt(
            ExifInterface.TAG_ORIENTATION,
            ExifInterface.ORIENTATION_NORMAL
        )
        return when (orientation) {
            ExifInterface.ORIENTATION_ROTATE_90 -> 90
            ExifInterface.ORIENTATION_ROTATE_180 -> 180
            ExifInterface.ORIENTATION_ROTATE_270 -> 270
            ExifInterface.ORIENTATION_FLIP_HORIZONTAL -> -1  // Special case
            ExifInterface.ORIENTATION_FLIP_VERTICAL -> -2    // Special case
            else -> 0
        }
    }

    private fun rotateBitmap(bitmap: Bitmap, degrees: Int): Bitmap {
        val matrix = Matrix()
        when (degrees) {
            -1 -> matrix.postScale(-1f, 1f)  // Flip horizontal
            -2 -> matrix.postScale(1f, -1f)  // Flip vertical
            else -> matrix.postRotate(degrees.toFloat())
        }
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true)
    }

    private fun saveBitmapToCache(bitmap: Bitmap): String? {
        return try {
            val file = File(cacheDir, "prediction_${System.currentTimeMillis()}.jpg")
            FileOutputStream(file).use { out ->
                bitmap.compress(Bitmap.CompressFormat.JPEG, 90, out)
            }
            file.absolutePath
        } catch (e: Exception) {
            LogCapture.e(TAG, "Failed to save bitmap: ${e.message}", e)
            null
        }
    }

    private fun displayResult(result: PredictionResult) {
        runOnUiThread {
            setLoading(false)

            // Show results, hide error
            binding.cardNoFace.visibility = View.GONE
            binding.resultsContainer.visibility = View.VISIBLE

            binding.tvAge.text = "${result.age} ans"
            binding.tvAgeConfidence.text = "${getString(R.string.confidence)}: ${result.ageConfidence.toPercentage()}"

            binding.tvGender.text = result.gender.label
            binding.tvGenderConfidence.text = "${getString(R.string.confidence)}: ${result.genderConfidence.toPercentage()}"

            binding.tvEthnicity.text = result.ethnicity.label
            binding.tvEthnicityConfidence.text = "${getString(R.string.confidence)}: ${result.ethnicityConfidence.toPercentage()}"

            binding.btnSave.isEnabled = true
        }
    }

    private fun showError(message: String) {
        runOnUiThread {
            setLoading(false)
            showToast(message)
            binding.tvAge.text = "-"
            binding.tvGender.text = "-"
            binding.tvEthnicity.text = "-"
            binding.tvAgeConfidence.text = ""
            binding.tvGenderConfidence.text = ""
            binding.tvEthnicityConfidence.text = ""
        }
    }

    private fun showNoFaceDetected() {
        runOnUiThread {
            setLoading(false)
            binding.cardNoFace.visibility = View.VISIBLE
            binding.resultsContainer.visibility = View.GONE
            binding.btnSave.isEnabled = false
        }
    }

    private fun savePrediction() {
        val result = currentResult ?: return
        val path = savedImagePath ?: imagePath ?: return
        val userId = authService.userId ?: return

        lifecycleScope.launch {
            firestoreRepository.savePrediction(
                userId = userId,
                imagePath = path,
                result = result
            ).onSuccess {
                showToast("Prédiction sauvegardée!")
                finish()
            }.onFailure { e ->
                LogCapture.e(TAG, "Failed to save prediction: ${e.message}", e)
                showToast("Erreur lors de la sauvegarde")
            }
        }
    }

    private fun setLoading(loading: Boolean) {
        runOnUiThread {
            binding.loadingOverlay.visibility = if (loading) View.VISIBLE else View.GONE
        }
    }

    override fun onDestroy() {
        super.onDestroy()
        try { predictorV2?.close() } catch (_: Exception) {}
        try { predictorV4?.close() } catch (_: Exception) {}
        try { predictorHybrid?.close() } catch (_: Exception) {}
        try { faceDetector?.close() } catch (_: Exception) {}
        predictorV2 = null
        predictorV4 = null
        predictorHybrid = null
        faceDetector = null
    }
}
