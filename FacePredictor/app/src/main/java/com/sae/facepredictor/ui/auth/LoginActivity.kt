package com.sae.facepredictor.ui.auth

import android.content.Intent
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ProgressBar
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.credentials.CredentialManager
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import androidx.credentials.GetCredentialResponse
import androidx.credentials.exceptions.GetCredentialException
import androidx.lifecycle.lifecycleScope
import com.google.android.libraries.identity.googleid.GetGoogleIdOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
import com.google.android.libraries.identity.googleid.GoogleIdTokenParsingException
import com.google.android.material.button.MaterialButton
import com.google.android.material.textfield.TextInputEditText
import com.sae.facepredictor.R
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.ui.main.MainActivity
import com.sae.facepredictor.utils.SecurityUtils
import com.sae.facepredictor.utils.showToast
import kotlinx.coroutines.launch

class LoginActivity : AppCompatActivity() {

    private lateinit var authService: FirebaseAuthService
    private lateinit var credentialManager: CredentialManager
    private lateinit var etEmail: TextInputEditText
    private lateinit var etPassword: TextInputEditText
    private lateinit var btnLogin: MaterialButton
    private lateinit var btnGoogleSignIn: MaterialButton
    private lateinit var tvRegister: TextView
    private lateinit var progressBar: ProgressBar

    companion object {
        private const val TAG = "LoginActivity"
        // Web Client ID depuis Firebase Console
        private const val WEB_CLIENT_ID = "619435245506-8c4uab1pjkddi13tsqj5iohph4mgjl35.apps.googleusercontent.com"
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_login)

        etEmail = findViewById(R.id.etEmail)
        etPassword = findViewById(R.id.etPassword)
        btnLogin = findViewById(R.id.btnLogin)
        btnGoogleSignIn = findViewById(R.id.btnGoogleSignIn)
        tvRegister = findViewById(R.id.tvRegister)
        progressBar = findViewById(R.id.progressBar)

        authService = FirebaseAuthService.getInstance()
        credentialManager = CredentialManager.create(this)

        // Check if already logged in
        if (authService.isLoggedIn) {
            navigateToMain()
            return
        }

        setupListeners()
    }

    private fun setupListeners() {
        // Login with email/password
        btnLogin.setOnClickListener {
            val email = etEmail.text.toString().trim()
            val password = etPassword.text.toString()

            if (validateInput(email, password)) {
                performLogin(email, password)
            }
        }

        // Google Sign-In
        btnGoogleSignIn.setOnClickListener {
            performGoogleSignIn()
        }

        // Navigate to register
        tvRegister.setOnClickListener {
            startActivity(Intent(this, RegisterActivity::class.java))
        }
    }

    private fun validateInput(email: String, password: String): Boolean {
        if (email.isEmpty() || password.isEmpty()) {
            showToast(getString(R.string.error_empty_fields))
            return false
        }

        if (!SecurityUtils.isValidEmail(email)) {
            showToast(getString(R.string.error_invalid_email))
            return false
        }

        return true
    }

    private fun performLogin(email: String, password: String) {
        setLoading(true)

        lifecycleScope.launch {
            val result = authService.login(email, password)

            result.onSuccess {
                navigateToMain()
            }.onFailure { error ->
                setLoading(false)
                showToast(error.message ?: getString(R.string.error_login))
            }
        }
    }

    private fun performGoogleSignIn() {
        setLoading(true)

        val googleIdOption = GetGoogleIdOption.Builder()
            .setFilterByAuthorizedAccounts(false)
            .setServerClientId(WEB_CLIENT_ID)
            .setAutoSelectEnabled(false)
            .build()

        val request = GetCredentialRequest.Builder()
            .addCredentialOption(googleIdOption)
            .build()

        lifecycleScope.launch {
            try {
                val result = credentialManager.getCredential(
                    request = request,
                    context = this@LoginActivity
                )
                handleGoogleSignInResult(result)
            } catch (e: GetCredentialException) {
                setLoading(false)
                Log.e(TAG, "Google Sign-In failed", e)
                showToast("Connexion Google annulée")
            }
        }
    }

    private fun handleGoogleSignInResult(result: GetCredentialResponse) {
        val credential = result.credential

        when (credential) {
            is CustomCredential -> {
                if (credential.type == GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_CREDENTIAL) {
                    try {
                        val googleIdTokenCredential = GoogleIdTokenCredential.createFrom(credential.data)
                        val idToken = googleIdTokenCredential.idToken

                        // Sign in with Firebase using the Google ID token
                        lifecycleScope.launch {
                            val firebaseResult = authService.signInWithGoogle(idToken)

                            firebaseResult.onSuccess {
                                navigateToMain()
                            }.onFailure { error ->
                                setLoading(false)
                                showToast(error.message ?: "Erreur de connexion Google")
                            }
                        }
                    } catch (e: GoogleIdTokenParsingException) {
                        setLoading(false)
                        Log.e(TAG, "Invalid Google ID token", e)
                        showToast("Erreur lors de la connexion Google")
                    }
                } else {
                    setLoading(false)
                    Log.e(TAG, "Unexpected credential type")
                    showToast("Type de credential non supporté")
                }
            }
            else -> {
                setLoading(false)
                Log.e(TAG, "Unexpected credential type")
                showToast("Type de credential non supporté")
            }
        }
    }

    private fun setLoading(loading: Boolean) {
        progressBar.visibility = if (loading) View.VISIBLE else View.GONE
        btnLogin.isEnabled = !loading
        btnGoogleSignIn.isEnabled = !loading
        etEmail.isEnabled = !loading
        etPassword.isEnabled = !loading
    }

    private fun navigateToMain() {
        startActivity(Intent(this, MainActivity::class.java))
        finish()
    }
}
