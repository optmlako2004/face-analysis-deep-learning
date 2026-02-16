package com.sae.facepredictor.ui.main

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.fragment.app.Fragment
import com.sae.facepredictor.R
import com.sae.facepredictor.databinding.FragmentSettingsBinding
import com.sae.facepredictor.utils.LogCapture
import com.sae.facepredictor.utils.PredictionMode
import com.sae.facepredictor.utils.SessionManager
import com.sae.facepredictor.utils.showToast

class SettingsFragment : Fragment() {

    companion object {
        private const val TAG = "SettingsFragment"
    }

    private var _binding: FragmentSettingsBinding? = null
    private val binding get() = _binding!!

    private lateinit var sessionManager: SessionManager

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentSettingsBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        sessionManager = SessionManager(requireContext())

        setupModelSwitch()
        setupListeners()
    }

    private fun setupModelSwitch() {
        // Initialize radio button state from preferences
        val currentMode = sessionManager.predictionMode
        when (currentMode) {
            PredictionMode.HYBRID -> binding.radioHybrid.isChecked = true
            PredictionMode.ORIENTED -> binding.radioOriented.isChecked = true
            PredictionMode.MULTITASK -> binding.radioMultitask.isChecked = true
            PredictionMode.TEST_MOBILENET -> binding.radioTestMobilenet.isChecked = true
        }

        // Handle radio button changes
        binding.radioGroupMode.setOnCheckedChangeListener { _, checkedId ->
            val newMode = when (checkedId) {
                R.id.radioHybrid -> PredictionMode.HYBRID
                R.id.radioOriented -> PredictionMode.ORIENTED
                R.id.radioMultitask -> PredictionMode.MULTITASK
                R.id.radioTestMobilenet -> PredictionMode.TEST_MOBILENET
                else -> PredictionMode.HYBRID
            }
            sessionManager.predictionMode = newMode
            LogCapture.i(TAG, "Model mode changed: ${newMode.label}")
            requireContext().showToast("Mode ${newMode.label} activé")
        }
    }

    private fun setupListeners() {
        binding.cardLogs.setOnClickListener {
            startActivity(Intent(requireContext(), LogsActivity::class.java))
        }
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}
