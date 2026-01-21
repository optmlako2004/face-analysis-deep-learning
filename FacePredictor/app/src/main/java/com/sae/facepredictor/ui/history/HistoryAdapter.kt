package com.sae.facepredictor.ui.history

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.load
import com.sae.facepredictor.data.firebase.FirestorePrediction
import com.sae.facepredictor.databinding.ItemHistoryBinding
import com.sae.facepredictor.utils.toFormattedDate
import java.io.File

class HistoryAdapter(
    private val onDeleteClick: (FirestorePrediction) -> Unit
) : ListAdapter<FirestorePrediction, HistoryAdapter.ViewHolder>(DiffCallback()) {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val binding = ItemHistoryBinding.inflate(
            LayoutInflater.from(parent.context),
            parent,
            false
        )
        return ViewHolder(binding)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        holder.bind(getItem(position))
    }

    inner class ViewHolder(
        private val binding: ItemHistoryBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(prediction: FirestorePrediction) {
            binding.tvDate.text = prediction.createdAtMillis.toFormattedDate()
            binding.tvAge.text = "Âge: ${prediction.predictedAge} ans"
            binding.tvGender.text = "Genre: ${prediction.predictedGender}"
            binding.tvEthnicity.text = "Ethnicité: ${prediction.predictedEthnicity}"

            // Load thumbnail
            val file = File(prediction.imagePath)
            if (file.exists()) {
                binding.ivThumbnail.load(file) {
                    crossfade(true)
                }
            }

            binding.btnDelete.setOnClickListener {
                onDeleteClick(prediction)
            }
        }
    }

    class DiffCallback : DiffUtil.ItemCallback<FirestorePrediction>() {
        override fun areItemsTheSame(oldItem: FirestorePrediction, newItem: FirestorePrediction): Boolean {
            return oldItem.id == newItem.id
        }

        override fun areContentsTheSame(oldItem: FirestorePrediction, newItem: FirestorePrediction): Boolean {
            return oldItem == newItem
        }
    }
}
