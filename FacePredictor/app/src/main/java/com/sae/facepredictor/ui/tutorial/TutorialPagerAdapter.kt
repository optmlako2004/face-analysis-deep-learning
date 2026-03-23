package com.sae.facepredictor.ui.tutorial

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.TextView
import androidx.annotation.DrawableRes
import androidx.annotation.StringRes
import androidx.recyclerview.widget.RecyclerView
import com.sae.facepredictor.R

data class TutorialPage(
    @DrawableRes val iconRes: Int,
    @StringRes val titleRes: Int,
    @StringRes val descriptionRes: Int
)

class TutorialPagerAdapter(
    private val pages: List<TutorialPage>
) : RecyclerView.Adapter<TutorialPagerAdapter.ViewHolder>() {

    class ViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val icon: ImageView = view.findViewById(R.id.tutorial_icon)
        val title: TextView = view.findViewById(R.id.tutorial_title)
        val description: TextView = view.findViewById(R.id.tutorial_description)
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_tutorial_page, parent, false)
        return ViewHolder(view)
    }

    override fun onBindViewHolder(holder: ViewHolder, position: Int) {
        val page = pages[position]
        holder.icon.setImageResource(page.iconRes)
        holder.title.setText(page.titleRes)
        holder.description.setText(page.descriptionRes)
    }

    override fun getItemCount(): Int = pages.size
}
