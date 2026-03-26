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
import com.sae.facepredictor.utils.toFormattedDate
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch

class HistoryActivity : AppCompatActivity() {

    private enum class AgeRange(val label: String) {
        ALL("Tous"),
        CHILD("0-17 ans"),
        YOUNG_ADULT("18-29 ans"),
        ADULT("30-49 ans"),
        SENIOR("50+ ans")
    }

    private lateinit var binding: ActivityHistoryBinding
    private lateinit var authService: FirebaseAuthService
    private lateinit var firestoreRepository: FirestoreRepository
    private lateinit var adapter: HistoryAdapter

    private var allPredictions: List<FirestorePrediction> = emptyList()
    private var selectedGender = "Tous"
    private var selectedAgeRange = AgeRange.ALL
    private var selectedEthnicity = "Toutes"

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

        binding.btnFilterGender.setOnClickListener {
            showGenderFilterDialog()
        }

        binding.btnFilterAgeRange.setOnClickListener {
            showAgeFilterDialog()
        }

        binding.btnFilterEthnicity.setOnClickListener {
            showEthnicityFilterDialog()
        }

        binding.btnClearFilters.setOnClickListener {
            clearFilters()
        }

        binding.rvHistory.apply {
            layoutManager = LinearLayoutManager(this@HistoryActivity)
            adapter = this@HistoryActivity.adapter
            setHasFixedSize(true)
        }

        updateFilterButtons()
    }

    private fun loadHistory() {
        val userId = authService.userId ?: return

        lifecycleScope.launch {
            try {
                firestoreRepository.getPredictionsByUser(userId)
                    .collectLatest { predictions ->
                        allPredictions = predictions
                        updateTopSummary(predictions)
                        applyFilters()
                    }
            } catch (e: Exception) {
                binding.progressBar.visibility = View.GONE
                binding.emptyState.visibility = View.VISIBLE
                binding.rvHistory.visibility = View.GONE
                binding.tvHistoryStatus.text = "Erreur"
                binding.tvLastAnalysis.text = "Dernière analyse : impossible à charger"
                showToast("Erreur de chargement de l'historique")
            }
        }
    }

    private fun updateTopSummary(predictions: List<FirestorePrediction>) {
        binding.tvLastAnalysis.text = predictions.firstOrNull()?.let {
            "Dernière analyse : ${it.createdAtMillis.toFormattedDate()}"
        } ?: "Dernière analyse : aucune donnée"
    }

    private fun applyFilters() {
        binding.progressBar.visibility = View.GONE

        val filteredPredictions = allPredictions.filter { prediction ->
            matchesGender(prediction) &&
                matchesAgeRange(prediction) &&
                matchesEthnicity(prediction)
        }

        binding.tvHistoryCount.text = filteredPredictions.size.toString()
        binding.tvHistoryStatus.text = if (filteredPredictions.isEmpty()) "Aucun résultat" else "Synchronisé"
        binding.tvSectionSubtitle.text = if (hasActiveFilters()) {
            "${getActiveFiltersLabel()} • ${filteredPredictions.size} résultat(s)"
        } else {
            "Chaque carte reprend l'image source et les attributs détectés."
        }
        binding.btnClearFilters.visibility = if (hasActiveFilters()) View.VISIBLE else View.GONE
        updateFilterButtons()

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
        return selectedGender == "Tous" ||
            prediction.predictedGender.equals(selectedGender, ignoreCase = true)
    }

    private fun matchesAgeRange(prediction: FirestorePrediction): Boolean {
        val age = prediction.predictedAge
        return when (selectedAgeRange) {
            AgeRange.ALL -> true
            AgeRange.CHILD -> age in 0..17
            AgeRange.YOUNG_ADULT -> age in 18..29
            AgeRange.ADULT -> age in 30..49
            AgeRange.SENIOR -> age >= 50
        }
    }

    private fun matchesEthnicity(prediction: FirestorePrediction): Boolean {
        return selectedEthnicity == "Toutes" ||
            prediction.predictedEthnicity.equals(selectedEthnicity, ignoreCase = true)
    }

    private fun showGenderFilterDialog() {
        val options = arrayOf("Tous", "Homme", "Femme")
        val checkedItem = options.indexOf(selectedGender).coerceAtLeast(0)

        AlertDialog.Builder(this)
            .setTitle("Filtrer par genre")
            .setSingleChoiceItems(options, checkedItem) { dialog, which ->
                selectedGender = options[which]
                applyFilters()
                dialog.dismiss()
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun showAgeFilterDialog() {
        val options = AgeRange.entries.toTypedArray()
        val labels = options.map { it.label }.toTypedArray()
        val checkedItem = options.indexOf(selectedAgeRange).coerceAtLeast(0)

        AlertDialog.Builder(this)
            .setTitle("Filtrer par tranche d'âge")
            .setSingleChoiceItems(labels, checkedItem) { dialog, which ->
                selectedAgeRange = options[which]
                applyFilters()
                dialog.dismiss()
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun showEthnicityFilterDialog() {
        val dynamicEthnicities = allPredictions
            .map { it.predictedEthnicity.trim() }
            .filter { it.isNotEmpty() }
            .distinct()
            .sorted()
        val options = listOf("Toutes") + dynamicEthnicities
        val checkedItem = options.indexOf(selectedEthnicity).let { if (it >= 0) it else 0 }

        AlertDialog.Builder(this)
            .setTitle("Filtrer par ethnie")
            .setSingleChoiceItems(options.toTypedArray(), checkedItem) { dialog, which ->
                selectedEthnicity = options[which]
                applyFilters()
                dialog.dismiss()
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun clearFilters() {
        selectedGender = "Tous"
        selectedAgeRange = AgeRange.ALL
        selectedEthnicity = "Toutes"
        applyFilters()
    }

    private fun hasActiveFilters(): Boolean {
        return selectedGender != "Tous" ||
            selectedAgeRange != AgeRange.ALL ||
            selectedEthnicity != "Toutes"
    }

    private fun updateFilterButtons() {
        binding.btnFilterGender.text = "Genre : $selectedGender"
        binding.btnFilterAgeRange.text = "Âge : ${selectedAgeRange.label}"
        binding.btnFilterEthnicity.text = "Ethnie : $selectedEthnicity"
    }

    private fun getActiveFiltersLabel(): String {
        val labels = mutableListOf<String>()
        if (selectedGender != "Tous") labels += selectedGender
        if (selectedAgeRange != AgeRange.ALL) labels += selectedAgeRange.label
        if (selectedEthnicity != "Toutes") labels += selectedEthnicity
        return labels.joinToString(" • ")
    }

    private fun showDeleteConfirmation(prediction: FirestorePrediction) {
        AlertDialog.Builder(this)
            .setTitle(getString(R.string.delete))
            .setMessage("Supprimer cette prédiction ?")
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
                    showToast("Supprimé")
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
                    showToast("Historique vidé")
                }
                .onFailure {
                    showToast("Erreur")
                }
        }
    }
}
