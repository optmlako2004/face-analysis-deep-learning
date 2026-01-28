package com.sae.facepredictor.ui.main

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.sae.facepredictor.R
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.data.firebase.FirestoreRepository
import com.sae.facepredictor.databinding.FragmentPredictionBinding
import com.sae.facepredictor.ui.camera.CameraActivity
import com.sae.facepredictor.ui.camera.RealtimeCameraActivity
import com.sae.facepredictor.ui.history.HistoryActivity
import com.sae.facepredictor.ui.prediction.PredictionResultActivity
import com.sae.facepredictor.utils.showToast
import kotlinx.coroutines.launch

class PredictionFragment : Fragment() {

    private var _binding: FragmentPredictionBinding? = null
    private val binding get() = _binding!!

    private lateinit var authService: FirebaseAuthService
    private lateinit var firestoreRepository: FirestoreRepository

    private var pendingRealtimeOpen = false

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
            if (pendingRealtimeOpen) {
                pendingRealtimeOpen = false
                openRealtimeCamera()
            } else {
                openCamera()
            }
        } else {
            pendingRealtimeOpen = false
            requireContext().showToast(getString(R.string.error_camera_permission))
        }
    }

    private val storagePermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            openGallery()
        } else {
            requireContext().showToast(getString(R.string.error_storage_permission))
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentPredictionBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        authService = FirebaseAuthService.getInstance()
        firestoreRepository = FirestoreRepository.getInstance()

        setupListeners()
    }

    override fun onResume() {
        super.onResume()
        loadHistoryCount()
    }

    private fun setupListeners() {
        binding.cardCamera.setOnClickListener {
            checkCameraPermission()
        }

        binding.cardRealtime.setOnClickListener {
            checkCameraPermissionForRealtime()
        }

        binding.cardGallery.setOnClickListener {
            checkStoragePermission()
        }

        binding.cardHistory.setOnClickListener {
            startActivity(Intent(requireContext(), HistoryActivity::class.java))
        }
    }

    private fun loadHistoryCount() {
        val userId = authService.userId ?: return

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val count = firestoreRepository.countPredictions(userId)
                binding.tvHistoryCount.text = "$count prédictions sauvegardées"
            } catch (e: Exception) {
                binding.tvHistoryCount.text = "0 prédictions sauvegardées"
            }
        }
    }

    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                openCamera()
            }
            else -> {
                cameraPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }

    private fun checkCameraPermissionForRealtime() {
        when {
            ContextCompat.checkSelfPermission(
                requireContext(),
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                openRealtimeCamera()
            }
            else -> {
                pendingRealtimeOpen = true
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
            ContextCompat.checkSelfPermission(requireContext(), permission) == PackageManager.PERMISSION_GRANTED -> {
                openGallery()
            }
            else -> {
                storagePermissionLauncher.launch(permission)
            }
        }
    }

    private fun openCamera() {
        startActivity(Intent(requireContext(), CameraActivity::class.java))
    }

    private fun openRealtimeCamera() {
        startActivity(Intent(requireContext(), RealtimeCameraActivity::class.java))
    }

    private fun openGallery() {
        galleryLauncher.launch("image/*")
    }

    private fun navigateToPrediction(imagePath: String) {
        val intent = Intent(requireContext(), PredictionResultActivity::class.java).apply {
            putExtra(PredictionResultActivity.EXTRA_IMAGE_PATH, imagePath)
        }
        startActivity(intent)
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
