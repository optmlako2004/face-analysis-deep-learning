package com.sae.facepredictor.ui.tutorial

import android.os.Bundle
import android.widget.ImageView
import android.widget.LinearLayout
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.viewpager2.widget.ViewPager2
import com.google.firebase.auth.FirebaseAuth
import com.sae.facepredictor.R
import com.sae.facepredictor.databinding.ActivityTutorialBinding
import com.sae.facepredictor.utils.SessionManager

class TutorialActivity : AppCompatActivity() {

    private lateinit var binding: ActivityTutorialBinding
    private lateinit var sessionManager: SessionManager

    private val pages = listOf(
        TutorialPage(
            R.drawable.ic_prediction,
            R.string.tutorial_welcome_title,
            R.string.tutorial_welcome_desc
        ),
        TutorialPage(
            R.drawable.ic_camera,
            R.string.tutorial_modes_title,
            R.string.tutorial_modes_desc
        ),
        TutorialPage(
            R.drawable.ic_face,
            R.string.tutorial_tips_title,
            R.string.tutorial_tips_desc
        ),
        TutorialPage(
            R.drawable.ic_settings,
            R.string.tutorial_ready_title,
            R.string.tutorial_ready_desc
        )
    )

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityTutorialBinding.inflate(layoutInflater)
        setContentView(binding.root)

        sessionManager = SessionManager(this)

        setupViewPager()
        setupDots()
        setupButtons()
    }

    private fun setupViewPager() {
        binding.viewPager.adapter = TutorialPagerAdapter(pages)
        binding.viewPager.registerOnPageChangeCallback(object : ViewPager2.OnPageChangeCallback() {
            override fun onPageSelected(position: Int) {
                updateDots(position)
                binding.btnNext.text = if (position == pages.size - 1) {
                    getString(R.string.tutorial_start)
                } else {
                    getString(R.string.tutorial_next)
                }
            }
        })
    }

    private fun setupDots() {
        for (i in pages.indices) {
            val dot = ImageView(this).apply {
                setImageDrawable(
                    ContextCompat.getDrawable(this@TutorialActivity, R.drawable.dot_inactive)
                )
                val params = LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.WRAP_CONTENT,
                    LinearLayout.LayoutParams.WRAP_CONTENT
                )
                params.setMargins(8, 0, 8, 0)
                layoutParams = params
            }
            binding.dotsLayout.addView(dot)
        }
        updateDots(0)
    }

    private fun updateDots(selectedPosition: Int) {
        for (i in 0 until binding.dotsLayout.childCount) {
            val dot = binding.dotsLayout.getChildAt(i) as ImageView
            dot.setImageDrawable(
                ContextCompat.getDrawable(
                    this,
                    if (i == selectedPosition) R.drawable.dot_active else R.drawable.dot_inactive
                )
            )
        }
    }

    private fun setupButtons() {
        binding.btnNext.setOnClickListener {
            val currentItem = binding.viewPager.currentItem
            if (currentItem < pages.size - 1) {
                binding.viewPager.currentItem = currentItem + 1
            } else {
                finishTutorial()
            }
        }

        binding.btnSkip.setOnClickListener {
            finishTutorial()
        }
    }

    private fun finishTutorial() {
        val userId = FirebaseAuth.getInstance().currentUser?.uid
        if (userId != null) {
            sessionManager.setHasSeenTutorial(userId, true)
        }
        finish()
    }
}
