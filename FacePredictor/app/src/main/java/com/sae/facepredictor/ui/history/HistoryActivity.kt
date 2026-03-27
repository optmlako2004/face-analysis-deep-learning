package com.sae.facepredictor.ui.history

import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AlertDialog
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import com.sae.facepredictor.R
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.data.firebase.FirestorePrediction
import com.sae.facepredictor.data.firebase.FirestoreRepository
import com.sae.facepredictor.databinding.ActivityHistoryBinding
import com.sae.facepredictor.utils.showToast
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class HistoryActivity : AppCompatActivity() {

    private lateinit var binding: ActivityHistoryBinding
    private lateinit var authService: FirebaseAuthService
    private lateinit var firestoreRepository: FirestoreRepository
    private lateinit var adapter: HistoryAdapter
    private var allPredictions: List<FirestorePrediction> = emptyList()
    private var selectedGender = GENDER_ALL
    private var selectedAgeRange = AGE_ALL
    private var selectedEthnicity = ETHNICITY_ALL

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityHistoryBinding.inflate(layoutInflater)
        setContentView(binding.root)

        authService = FirebaseAuthService.getInstance()
        firestoreRepository = FirestoreRepository.getInstance()

        setupUI()
        loadHistory()
    }

    private fun setupUI() {
        binding.toolbar.setNavigationOnClickListener {
            finish()
        }

        binding.toolbar.setOnMenuItemClickListener { menuItem ->
            when (menuItem.itemId) {
                R.id.action_clear_history -> {
                    showClearConfirmation()
                    true
                }
                else -> false
            }
        }

        adapter = HistoryAdapter(
            onDeleteClick = { prediction ->
                showDeleteConfirmation(prediction)
            }
        )

        binding.rvHistory.apply {
            layoutManager = LinearLayoutManager(this@HistoryActivity)
            adapter = this@HistoryActivity.adapter
            setHasFixedSize(true)
        }

        binding.btnGenderFilter.setOnClickListener {
            showSingleChoiceDialog(
                title = "Filtrer par genre",
                options = GENDER_OPTIONS,
                selectedValue = selectedGender
            ) {
                selectedGender = it
                applyFilters()
            }
        }

        binding.btnAgeFilter.setOnClickListener {
            showSingleChoiceDialog(
                title = "Filtrer par age",
                options = AGE_OPTIONS,
                selectedValue = selectedAgeRange
            ) {
                selectedAgeRange = it
                applyFilters()
            }
        }

        binding.btnEthnicityFilter.setOnClickListener {
            val options = buildEthnicityOptions()
            showSingleChoiceDialog(
                title = "Filtrer par ethnie",
                options = options,
                selectedValue = selectedEthnicity
            ) {
                selectedEthnicity = it
                applyFilters()
            }
        }

        binding.btnResetFilters.setOnClickListener {
            selectedGender = GENDER_ALL
            selectedAgeRange = AGE_ALL
            selectedEthnicity = ETHNICITY_ALL
            applyFilters()
        }

        updateFilterUI()
    }

    private fun loadHistory() {
        val userId = authService.userId ?: return

        lifecycleScope.launch {
            try {
                firestoreRepository.getPredictionsByUser(userId)
                    .collectLatest { predictions ->
                        allPredictions = predictions
                        binding.progressBar.visibility = View.GONE
                        applyFilters()
                    }
            } catch (e: Exception) {
                binding.progressBar.visibility = View.GONE
                binding.emptyState.visibility = View.VISIBLE
                binding.rvHistory.visibility = View.GONE
                showToast("Erreur de chargement de l'historique")
            }
        }
    }

    private fun applyFilters() {
        val filteredPredictions = allPredictions.filter { prediction ->
            matchesGender(prediction) &&
                matchesAgeRange(prediction) &&
                matchesEthnicity(prediction)
        }

        updateFilterUI(filteredPredictions.size)

        if (filteredPredictions.isEmpty()) {
            binding.emptyState.visibility = View.VISIBLE
            binding.rvHistory.visibility = View.GONE
        } else {
            binding.emptyState.visibility = View.GONE
            binding.rvHistory.visibility = View.VISIBLE
            adapter.submitList(filteredPredictions)
        }
    }

    private fun matchesGender(prediction: FirestorePrediction): Boolean {
        if (selectedGender == GENDER_ALL) return true
        return prediction.predictedGender.equals(selectedGender, ignoreCase = true)
    }

    private fun matchesAgeRange(prediction: FirestorePrediction): Boolean {
        return when (selectedAgeRange) {
            AGE_ALL -> true
            "0-17" -> prediction.predictedAge in 0..17
            "18-29" -> prediction.predictedAge in 18..29
            "30-49" -> prediction.predictedAge in 30..49
            "50+" -> prediction.predictedAge >= 50
            else -> true
        }
    }

    private fun matchesEthnicity(prediction: FirestorePrediction): Boolean {
        if (selectedEthnicity == ETHNICITY_ALL) return true
        return prediction.predictedEthnicity.equals(selectedEthnicity, ignoreCase = true)
    }

    private fun updateFilterUI(resultCount: Int = allPredictions.size) {
        binding.btnGenderFilter.text = "Genre : $selectedGender"
        binding.btnAgeFilter.text = "Age : $selectedAgeRange"
        binding.btnEthnicityFilter.text = "Ethnie : $selectedEthnicity"

        val hasActiveFilters = selectedGender != GENDER_ALL ||
            selectedAgeRange != AGE_ALL ||
            selectedEthnicity != ETHNICITY_ALL

        binding.btnResetFilters.visibility = if (hasActiveFilters) View.VISIBLE else View.GONE
        binding.tvFilterSummary.text = if (hasActiveFilters) {
            "$resultCount resultat(s) apres filtrage"
        } else {
            getString(R.string.history_summary_all)
        }
    }

    private fun buildEthnicityOptions(): Array<String> {
        val dynamicEthnicities = allPredictions
            .map { it.predictedEthnicity.trim() }
            .filter { it.isNotEmpty() }
            .distinct()
            .sorted()

        return arrayOf(ETHNICITY_ALL, *dynamicEthnicities.toTypedArray())
    }

    private fun showSingleChoiceDialog(
        title: String,
        options: Array<String>,
        selectedValue: String,
        onSelected: (String) -> Unit
    ) {
        val selectedIndex = options.indexOf(selectedValue).coerceAtLeast(0)
        AlertDialog.Builder(this)
            .setTitle(title)
            .setSingleChoiceItems(options, selectedIndex) { dialog, which ->
                onSelected(options[which])
                dialog.dismiss()
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun showDeleteConfirmation(prediction: FirestorePrediction) {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.delete))
            .setMessage("Supprimer cette prediction ?")
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                deletePrediction(prediction)
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun showClearConfirmation() {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.clear_history))
            .setMessage("Supprimer tout l'historique ?")
            .setPositiveButton(getString(R.string.delete)) { _, _ ->
                clearHistory()
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun deletePrediction(prediction: FirestorePrediction) {
        lifecycleScope.launch {
            firestoreRepository.deletePrediction(prediction.id)
                .onSuccess {
                    showToast("Supprime")
                }
                .onFailure {
                    showToast("Erreur de suppression")
                }
        }
    }

    private fun clearHistory() {
        val userId = authService.userId ?: return

        lifecycleScope.launch {
            firestoreRepository.clearHistory(userId)
                .onSuccess {
                    showToast("Historique vide")
                }
                .onFailure {
                    showToast("Erreur")
                }
        }
    }

    companion object {
        private const val GENDER_ALL = "Tous"
        private const val AGE_ALL = "Tous"
        private const val ETHNICITY_ALL = "Toutes"

        private val GENDER_OPTIONS = arrayOf(GENDER_ALL, "Homme", "Femme")
        private val AGE_OPTIONS = arrayOf(AGE_ALL, "0-17", "18-29", "30-49", "50+")
    }
}
