package com.sae.facepredictor.ui.main

import android.content.ClipData
import android.content.ClipboardManager
import android.content.Context
import android.os.Bundle
import android.widget.ScrollView
import androidx.appcompat.app.AppCompatActivity
import com.sae.facepredictor.databinding.ActivityLogsBinding
import com.sae.facepredictor.utils.LogCapture
import com.sae.facepredictor.utils.showToast

class LogsActivity : AppCompatActivity(), LogCapture.LogListener {

    companion object {
        private const val TAG = "LogsActivity"
    }

    private lateinit var binding: ActivityLogsBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityLogsBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        setupListeners()
        refreshLogs()
    }

    override fun onResume() {
        super.onResume()
        LogCapture.setLogListener(this)
    }

    override fun onPause() {
        super.onPause()
        LogCapture.setLogListener(null)
    }

    override fun onNewLog(log: String) {
        runOnUiThread {
            val currentText = binding.tvLogs.text.toString()
            val newText = if (currentText == "Chargement des logs...") {
                log
            } else {
                "$currentText\n$log"
            }
            binding.tvLogs.text = newText
            scrollToBottom()
        }
    }

    private fun setupUI() {
        binding.toolbar.setNavigationOnClickListener {
            finish()
        }
    }

    private fun setupListeners() {
        binding.btnCopyLogs.setOnClickListener {
            val logs = binding.tvLogs.text.toString()
            val clipboard = getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
            val clip = ClipData.newPlainText("FacePredictor Logs", logs)
            clipboard.setPrimaryClip(clip)
            showToast("Logs copiés!")
        }

        binding.btnRefreshLogs.setOnClickListener {
            refreshLogs()
            showToast("Logs actualisés")
        }

        binding.btnClearLogs.setOnClickListener {
            LogCapture.clear()
            binding.tvLogs.text = "Logs effacés."
            LogCapture.i(TAG, "Logs cleared by user")
        }
    }

    private fun refreshLogs() {
        val appLogs = LogCapture.getLogsAsString()
        val logcatLogs = LogCapture.getRecentLogcat()

        val displayLogs = buildString {
            append("═══════════════════════════════════\n")
            append("         APP LOGS\n")
            append("═══════════════════════════════════\n\n")
            if (appLogs.isNotEmpty()) {
                append(appLogs)
            } else {
                append("Aucun log applicatif pour le moment.")
            }
            append("\n\n")
            append("═══════════════════════════════════\n")
            append("    LOGCAT (100 dernières lignes)\n")
            append("═══════════════════════════════════\n\n")
            append(logcatLogs)
        }

        binding.tvLogs.text = displayLogs
        scrollToBottom()
    }

    private fun scrollToBottom() {
        binding.scrollLogs.post {
            binding.scrollLogs.fullScroll(ScrollView.FOCUS_DOWN)
        }
    }
}
