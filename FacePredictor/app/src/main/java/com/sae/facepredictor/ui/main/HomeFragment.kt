package com.sae.facepredictor.ui.main

import android.net.Uri
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.MediaController
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.bottomnavigation.BottomNavigationView
import com.sae.facepredictor.R
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.data.firebase.FirestoreRepository
import com.sae.facepredictor.databinding.FragmentHomeBinding
import kotlinx.coroutines.launch

class HomeFragment : Fragment() {

    private var _binding: FragmentHomeBinding? = null
    private val binding get() = _binding!!

    private lateinit var authService: FirebaseAuthService
    private lateinit var firestoreRepository: FirestoreRepository
    private var videoReady = false

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentHomeBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        authService = FirebaseAuthService.getInstance()
        firestoreRepository = FirestoreRepository.getInstance()

        setupVideo()
        setupListeners()
        loadStats()
    }

    private fun setupVideo() {
        try {
            val videoUri = Uri.parse("android.resource://${requireContext().packageName}/${R.raw.demo_video}")
            binding.videoDemo.setVideoURI(videoUri)

            val mediaController = MediaController(requireContext())
            mediaController.setAnchorView(binding.videoDemo)
            binding.videoDemo.setMediaController(mediaController)

            binding.videoDemo.setOnPreparedListener { mp ->
                mp.isLooping = true
                videoReady = true
            }

            // Play button: start video and hide play icon
            binding.ivPlayIcon.setOnClickListener {
                if (videoReady) {
                    binding.videoDemo.start()
                    binding.ivPlayIcon.visibility = View.GONE
                }
            }

            // Show play icon again when video completes (if not looping)
            binding.videoDemo.setOnCompletionListener {
                binding.ivPlayIcon.visibility = View.VISIBLE
            }
        } catch (e: Exception) {
            binding.videoDemo.visibility = View.GONE
            binding.ivPlayIcon.visibility = View.GONE
        }
    }

    private fun setupListeners() {
        binding.btnGetStarted.setOnClickListener {
            requireActivity().findViewById<BottomNavigationView>(R.id.bottom_navigation)
                .selectedItemId = R.id.navigation_prediction
        }
    }

    private fun loadStats() {
        val userId = authService.userId ?: return

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val count = firestoreRepository.countPredictions(userId).getOrDefault(0)
                binding.tvPredictionCount.text = count.toString()
            } catch (e: Exception) {
                binding.tvPredictionCount.text = "0"
            }
        }
    }

    override fun onPause() {
        super.onPause()
        try {
            if (binding.videoDemo.isPlaying) {
                binding.videoDemo.pause()
            }
        } catch (_: Exception) {}
    }

    override fun onDestroyView() {
        super.onDestroyView()
        try { binding.videoDemo.stopPlayback() } catch (_: Exception) {}
        _binding = null
    }
}