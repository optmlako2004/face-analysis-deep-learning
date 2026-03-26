package com.sae.facepredictor.ui.history

import android.content.res.ColorStateList
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
        holder.bind(getItem(position), position)
    }

    inner class ViewHolder(
        private val binding: ItemHistoryBinding
    ) : RecyclerView.ViewHolder(binding.root) {

        fun bind(prediction: FirestorePrediction, position: Int) {
            binding.tvDate.text = prediction.createdAtMillis.toFormattedDate()
            binding.tvAge.text = "${prediction.predictedAge} ans"
            binding.tvGender.text = prediction.predictedGender
            binding.tvEthnicity.text = prediction.predictedEthnicity
            binding.tvCardLabel.text = if (position == 0) "Récent" else "Analyse"

            val file = File(prediction.imagePath)
            binding.ivThumbnail.load(file) {
                crossfade(true)
                placeholder(R.drawable.ic_face)
                error(R.drawable.ic_face)
            }

            val context = binding.root.context
            val isFemale = prediction.predictedGender.equals("Femme", ignoreCase = true)
            val genderTextColor = if (isFemale) {
                context.getColor(R.color.female_color)
            } else {
                context.getColor(R.color.male_color)
            }
            val genderBackgroundColor = if (isFemale) {
                context.getColor(R.color.female_background)
            } else {
                context.getColor(R.color.male_background)
            }

            binding.tvGender.setTextColor(genderTextColor)
            binding.tvGender.backgroundTintList = ColorStateList.valueOf(genderBackgroundColor)
            binding.tvEthnicity.backgroundTintList = ColorStateList.valueOf(
                context.getColor(R.color.secondary_container)
            )
            binding.tvCardLabel.backgroundTintList = ColorStateList.valueOf(
                context.getColor(R.color.primary_container)
            )

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
