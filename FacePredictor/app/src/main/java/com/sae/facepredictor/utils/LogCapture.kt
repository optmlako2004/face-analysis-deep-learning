package com.sae.facepredictor.utils

import android.util.Log
import java.io.BufferedReader
import java.io.InputStreamReader
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.CopyOnWriteArrayList

/**
 * Utility class to capture application logs for debugging purposes.
 * This is a temporary debugging tool.
 */
object LogCapture {

    private val logs = CopyOnWriteArrayList<String>()
    private val dateFormat = SimpleDateFormat("HH:mm:ss.SSS", Locale.getDefault())
    private var listener: LogListener? = null

    interface LogListener {
        fun onNewLog(log: String)
    }

    fun setLogListener(listener: LogListener?) {
        this.listener = listener
    }

    fun log(tag: String, message: String, level: String = "D") {
        val timestamp = dateFormat.format(Date())
        val logEntry = "[$timestamp] $level/$tag: $message"
        logs.add(logEntry)
        listener?.onNewLog(logEntry)

        // Keep only last 500 logs to avoid memory issues
        while (logs.size > 500) {
            logs.removeAt(0)
        }
    }

    fun d(tag: String, message: String) {
        Log.d(tag, message)
        log(tag, message, "D")
    }

    fun e(tag: String, message: String, throwable: Throwable? = null) {
        Log.e(tag, message, throwable)
        val fullMessage = if (throwable != null) {
            "$message\n${throwable.stackTraceToString()}"
        } else {
            message
        }
        log(tag, fullMessage, "E")
    }

    fun w(tag: String, message: String) {
        Log.w(tag, message)
        log(tag, message, "W")
    }

    fun i(tag: String, message: String) {
        Log.i(tag, message)
        log(tag, message, "I")
    }

    fun getAllLogs(): List<String> = logs.toList()

    fun getLogsAsString(): String = logs.joinToString("\n")

    fun clear() {
        logs.clear()
    }

    /**
     * Capture logcat output from the current process
     */
    fun captureLogcat(): String {
        return try {
            val process = Runtime.getRuntime().exec("logcat -d -v time *:V")
            val bufferedReader = BufferedReader(InputStreamReader(process.inputStream))
            val log = StringBuilder()
            var line: String?
            while (bufferedReader.readLine().also { line = it } != null) {
                log.append(line).append("\n")
            }
            log.toString()
        } catch (e: Exception) {
            "Error capturing logcat: ${e.message}"
        }
    }

    /**
     * Get recent logcat entries (last 100 lines)
     */
    fun getRecentLogcat(): String {
        return try {
            val process = Runtime.getRuntime().exec("logcat -d -t 100 *:V")
            val bufferedReader = BufferedReader(InputStreamReader(process.inputStream))
            val log = StringBuilder()
            var line: String?
            while (bufferedReader.readLine().also { line = it } != null) {
                log.append(line).append("\n")
            }
            log.toString()
        } catch (e: Exception) {
            "Error capturing logcat: ${e.message}"
        }
    }
}
