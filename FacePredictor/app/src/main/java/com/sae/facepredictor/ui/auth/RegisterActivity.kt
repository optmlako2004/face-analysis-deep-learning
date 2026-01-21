package com.sae.facepredictor.ui.auth

import android.os.Bundle
import android.view.View
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.sae.facepredictor.R
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.databinding.ActivityRegisterBinding
import com.sae.facepredictor.utils.SecurityUtils
import com.sae.facepredictor.utils.showToast
import kotlinx.coroutines.launch

class RegisterActivity : AppCompatActivity() {

    private lateinit var binding: ActivityRegisterBinding
    private lateinit var authService: FirebaseAuthService

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityRegisterBinding.inflate(layoutInflater)
        setContentView(binding.root)

        authService = FirebaseAuthService.getInstance()

        setupListeners()
    }

    private fun setupListeners() {
        binding.btnRegister.setOnClickListener {
            val username = binding.etUsername.text.toString().trim()
            val email = binding.etEmail.text.toString().trim()
            val password = binding.etPassword.text.toString()
            val confirmPassword = binding.etConfirmPassword.text.toString()

            if (validateInput(username, email, password, confirmPassword)) {
                performRegistration(username, email, password)
            }
        }

        binding.tvLogin.setOnClickListener {
            finish()
        }
    }

    private fun validateInput(
        username: String,
        email: String,
        password: String,
        confirmPassword: String
    ): Boolean {
        if (username.isEmpty() || email.isEmpty() || password.isEmpty() || confirmPassword.isEmpty()) {
            showToast(getString(R.string.error_empty_fields))
            return false
        }

        if (!SecurityUtils.isValidEmail(email)) {
            showToast(getString(R.string.error_invalid_email))
            return false
        }

        if (!SecurityUtils.isValidPassword(password)) {
            showToast("Le mot de passe doit contenir au moins 6 caractères")
            return false
        }

        if (password != confirmPassword) {
            showToast(getString(R.string.error_password_mismatch))
            return false
        }

        return true
    }

    private fun performRegistration(username: String, email: String, password: String) {
        setLoading(true)

        lifecycleScope.launch {
            val result = authService.register(username, email, password)

            result.onSuccess {
                showToast("Inscription réussie!")
                finish()
            }.onFailure { error ->
                setLoading(false)
                showToast(error.message ?: getString(R.string.error_register))
            }
        }
    }

    private fun setLoading(loading: Boolean) {
        binding.progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        binding.btnRegister.isEnabled = !loading
        binding.etUsername.isEnabled = !loading
        binding.etEmail.isEnabled = !loading
        binding.etPassword.isEnabled = !loading
        binding.etConfirmPassword.isEnabled = !loading
    }
}
