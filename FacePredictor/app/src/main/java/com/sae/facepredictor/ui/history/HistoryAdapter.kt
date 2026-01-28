package com.sae.facepredictor.ui.history

import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.DiffUtil
import androidx.recyclerview.widget.ListAdapter
import androidx.recyclerview.widget.RecyclerView
import coil.load
import com.sae.facepredictor.R
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
            // Date
            binding.tvDate.text = prediction.createdAtMillis.toFormattedDate()

            // Age chip
            binding.tvAge.text = "${prediction.predictedAge} ans"

            // Gender chip
            binding.tvGender.text = prediction.predictedGender

            // Ethnicity
            binding.tvEthnicity.text = prediction.predictedEthnicity

            // Load thumbnail with placeholder to prevent layout jumps
            val file = File(prediction.imagePath)
            binding.ivThumbnail.load(file) {
                crossfade(true)
                placeholder(R.drawable.ic_face)
                error(R.drawable.ic_face)
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
