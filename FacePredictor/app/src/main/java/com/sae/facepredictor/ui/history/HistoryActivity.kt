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
    }

    private fun loadHistory() {
        val userId = authService.userId ?: return

        lifecycleScope.launch {
            try {
                firestoreRepository.getPredictionsByUser(userId)
                    .collectLatest { predictions ->
                        updateUI(predictions)
                    }
            } catch (e: Exception) {
                binding.progressBar.visibility = View.GONE
                binding.emptyState.visibility = View.VISIBLE
                binding.rvHistory.visibility = View.GONE
                showToast("Erreur de chargement de l'historique")
            }
        }
    }

    private fun updateUI(predictions: List<FirestorePrediction>) {
        binding.progressBar.visibility = View.GONE

        if (predictions.isEmpty()) {
            binding.emptyState.visibility = View.VISIBLE
            binding.rvHistory.visibility = View.GONE
        } else {
            binding.emptyState.visibility = View.GONE
            binding.rvHistory.visibility = View.VISIBLE
            adapter.submitList(predictions)
        }
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
