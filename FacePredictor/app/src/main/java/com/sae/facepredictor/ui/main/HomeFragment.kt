package com.sae.facepredictor.ui.main

import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import androidx.navigation.fragment.findNavController
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

        setupListeners()
        loadStats()
    }

    private fun setupListeners() {
        binding.btnGetStarted.setOnClickListener {
            // Navigate to prediction tab
            findNavController().navigate(R.id.navigation_prediction)
        }

        binding.cardVideo.setOnClickListener {
            // TODO: Open video when available
        }
    }

    private fun loadStats() {
        val userId = authService.userId ?: return

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val count = firestoreRepository.countPredictions(userId)
                binding.tvPredictionCount.text = count.toString()
            } catch (e: Exception) {
                binding.tvPredictionCount.text = "0"
            }
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
