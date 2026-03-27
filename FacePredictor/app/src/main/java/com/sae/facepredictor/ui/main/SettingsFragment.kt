package com.sae.facepredictor.ui.main

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AppCompatDelegate
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
        setupDarkMode()
        setupListeners()
    }

    private fun setupModelSwitch() {
        selectMode(sessionManager.predictionMode)

        binding.radioHybrid.setOnClickListener { selectMode(PredictionMode.HYBRID) }
        binding.radioOriented.setOnClickListener { selectMode(PredictionMode.ORIENTED) }
        binding.radioMultitask.setOnClickListener { selectMode(PredictionMode.MULTITASK) }
        binding.radioMoe.setOnClickListener { selectMode(PredictionMode.MOE) }
    }

    private fun selectMode(mode: PredictionMode) {
        binding.radioHybrid.isChecked = false
        binding.radioOriented.isChecked = false
        binding.radioMultitask.isChecked = false
        binding.radioMoe.isChecked = false

        when (mode) {
            PredictionMode.HYBRID -> binding.radioHybrid.isChecked = true
            PredictionMode.ORIENTED -> binding.radioOriented.isChecked = true
            PredictionMode.MULTITASK -> binding.radioMultitask.isChecked = true
            PredictionMode.MOE -> binding.radioMoe.isChecked = true
        }

        if (sessionManager.predictionMode != mode) {
            sessionManager.predictionMode = mode
            LogCapture.i(TAG, "Model mode changed: ${mode.label}")
            requireContext().showToast("Mode ${mode.label} activé")
        }
    }

    private fun setupDarkMode() {
        val currentMode = sessionManager.darkMode

        // Set switch state: -1 (system) and 0 (light) = off, 1 (dark) = on
        binding.switchDarkMode.isChecked = currentMode == 1
        updateDarkModeDesc(currentMode)

        binding.switchDarkMode.setOnCheckedChangeListener { _, isChecked ->
            val newMode = if (isChecked) 1 else 0
            sessionManager.darkMode = newMode
            updateDarkModeDesc(newMode)

            AppCompatDelegate.setDefaultNightMode(
                if (isChecked) AppCompatDelegate.MODE_NIGHT_YES
                else AppCompatDelegate.MODE_NIGHT_NO
            )

            LogCapture.i(TAG, "Dark mode: ${if (isChecked) "ON" else "OFF"}")
        }
    }

    private fun updateDarkModeDesc(mode: Int) {
        binding.tvDarkModeDesc.text = when (mode) {
            1 -> getString(R.string.settings_dark_mode_on)
            0 -> getString(R.string.settings_dark_mode_off)
            else -> getString(R.string.settings_dark_mode_system)
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