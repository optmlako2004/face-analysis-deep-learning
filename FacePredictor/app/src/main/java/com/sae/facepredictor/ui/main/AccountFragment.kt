package com.sae.facepredictor.ui.main

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.EditText
import android.widget.FrameLayout
import android.widget.ImageView
import android.widget.TextView
import androidx.credentials.CredentialManager
import androidx.credentials.CustomCredential
import androidx.credentials.GetCredentialRequest
import androidx.credentials.exceptions.GetCredentialException
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.libraries.identity.googleid.GetGoogleIdOption
import com.google.android.libraries.identity.googleid.GoogleIdTokenCredential
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import com.google.firebase.auth.EmailAuthProvider
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.GoogleAuthProvider
import com.google.firebase.auth.UserProfileChangeRequest
import com.sae.facepredictor.R
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.data.firebase.FirestoreRepository
import com.sae.facepredictor.databinding.FragmentAccountBinding
import com.sae.facepredictor.ui.auth.LoginActivity
import com.sae.facepredictor.utils.LogCapture
import com.sae.facepredictor.utils.showToast
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await
import org.json.JSONArray
import org.json.JSONObject
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class AccountFragment : Fragment() {

    companion object {
        private const val TAG = "AccountFragment"
        private const val WEB_CLIENT_ID = "619435245506-8c4uab1pjkddi13tsqj5iohph4mgjl35.apps.googleusercontent.com"
    }

    private var _binding: FragmentAccountBinding? = null
    private val binding get() = _binding!!

    private lateinit var authService: FirebaseAuthService
    private lateinit var firestoreRepository: FirestoreRepository

    private var isGoogleUser = false

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentAccountBinding.inflate(inflater, container, false)
        return binding.root
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)

        authService = FirebaseAuthService.getInstance()
        firestoreRepository = FirestoreRepository.getInstance()

        setupUserInfo()
        setupSecurityInfo()
        setupStats()
        setupListeners()
    }

    private fun setupUserInfo() {
        val user = FirebaseAuth.getInstance().currentUser ?: return

        val displayName = user.displayName
            ?: user.email?.substringBefore("@")
            ?: "Utilisateur"
        binding.tvUserName.text = displayName
        binding.tvUserEmail.text = user.email ?: "Email non disponible"

        // Avatar initials
        val initials = displayName.split(" ")
            .take(2)
            .mapNotNull { it.firstOrNull()?.uppercase() }
            .joinToString("")
            .ifEmpty { "?" }
        binding.tvAvatarInitials.text = initials

        // Check if Google user
        isGoogleUser = user.providerData.any { it.providerId == GoogleAuthProvider.PROVIDER_ID }

        if (isGoogleUser) {
            binding.tvProvider.text = getString(R.string.account_provider_google)
            binding.tvPasswordAction.text = getString(R.string.account_set_password)
            binding.tvPasswordDesc.visibility = View.VISIBLE
            binding.tvPasswordDesc.text = getString(R.string.account_set_password_desc)
        } else {
            binding.tvProvider.text = getString(R.string.account_provider_email)
            binding.tvPasswordAction.text = getString(R.string.account_change_password)
            binding.tvPasswordDesc.visibility = View.GONE
        }
    }

    private fun setupSecurityInfo() {
        val user = FirebaseAuth.getInstance().currentUser ?: return
        val dateFormat = SimpleDateFormat("d MMM yyyy", Locale.FRANCE)

        // Membre depuis
        user.metadata?.creationTimestamp?.let { ts ->
            binding.tvMemberSince.text = dateFormat.format(Date(ts))
        }

        // Derniere connexion
        user.metadata?.lastSignInTimestamp?.let { ts ->
            val lastLogin = Date(ts)
            val now = Date()
            val diffMs = now.time - lastLogin.time
            val diffHours = diffMs / (1000 * 60 * 60)

            binding.tvLastLogin.text = when {
                diffHours < 1 -> "Maintenant"
                diffHours < 24 -> "Il y a ${diffHours}h"
                diffHours < 48 -> "Hier"
                else -> dateFormat.format(lastLogin)
            }
        }

        // Email verifie
        val verified = user.isEmailVerified || isGoogleUser
        binding.tvEmailVerified.text = if (verified) "Oui" else "Non"
        binding.tvEmailVerified.setTextColor(
            resources.getColor(
                if (verified) R.color.success else R.color.warning,
                null
            )
        )

        // Methodes d'authentification
        val methods = mutableListOf<String>()
        user.providerData.forEach { profile ->
            when (profile.providerId) {
                GoogleAuthProvider.PROVIDER_ID -> methods.add("Google")
                EmailAuthProvider.PROVIDER_ID -> methods.add("Email")
            }
        }
        binding.tvAuthMethods.text = methods.distinct().joinToString(" + ")
    }

    private fun setupStats() {
        val user = FirebaseAuth.getInstance().currentUser ?: return

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val result = firestoreRepository.getPredictionsByUserOnce(user.uid)
                val predictions = result.getOrDefault(emptyList())

                // Total
                binding.tvStatTotal.text = predictions.size.toString()

                if (predictions.isNotEmpty()) {
                    // Age moyen
                    val avgAge = predictions.map { it.predictedAge }.average()
                    binding.tvStatAvgAge.text = String.format("%.0f", avgAge)

                    // Genre le plus frequent
                    val genderCounts = predictions.groupingBy { it.predictedGender }.eachCount()
                    val topGender = genderCounts.maxByOrNull { it.value }?.key ?: "—"
                    binding.tvStatTopGender.text = when (topGender) {
                        "MALE" -> "H"
                        "FEMALE" -> "F"
                        else -> topGender.take(1)
                    }

                    // Derniere prediction
                    val lastPrediction = predictions.firstOrNull()
                    lastPrediction?.createdAt?.let { ts ->
                        val dateFormat = SimpleDateFormat("d MMM yyyy, HH:mm", Locale.FRANCE)
                        binding.tvStatLastPrediction.text =
                            getString(R.string.account_stats_last, dateFormat.format(ts.toDate()))
                    }
                }
            } catch (e: Exception) {
                LogCapture.e(TAG, "Failed to load stats: ${e.message}", e)
            }
        }
    }

    private fun setupListeners() {
        binding.cardChangePassword.setOnClickListener {
            if (isGoogleUser) showSetPasswordDialog() else showChangePasswordDialog()
        }

        binding.cardDeleteAccount.setOnClickListener {
            showDeleteAccountConfirmation()
        }

        binding.btnLogout.setOnClickListener {
            showLogoutConfirmation()
        }

        binding.btnEditName.setOnClickListener {
            showEditNameDialog()
        }

        binding.cardExportData.setOnClickListener {
            exportData()
        }
    }

    private fun showEditNameDialog() {
        val user = FirebaseAuth.getInstance().currentUser ?: return
        val currentName = user.displayName ?: ""

        val inputLayout = TextInputLayout(requireContext()).apply {
            setPadding(50, 20, 50, 0)
            hint = getString(R.string.account_edit_name_hint)
        }
        val editText = TextInputEditText(inputLayout.context).apply {
            setText(currentName)
        }
        inputLayout.addView(editText)

        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.account_edit_name)
            .setView(inputLayout)
            .setPositiveButton("Enregistrer") { _, _ ->
                val newName = editText.text.toString().trim()
                if (newName.isNotEmpty() && newName != currentName) {
                    updateDisplayName(newName)
                }
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun updateDisplayName(newName: String) {
        val user = FirebaseAuth.getInstance().currentUser ?: return

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val profileUpdates = UserProfileChangeRequest.Builder()
                    .setDisplayName(newName)
                    .build()
                user.updateProfile(profileUpdates).await()

                binding.tvUserName.text = newName
                val initials = newName.split(" ")
                    .take(2)
                    .mapNotNull { it.firstOrNull()?.uppercase() }
                    .joinToString("")
                    .ifEmpty { "?" }
                binding.tvAvatarInitials.text = initials

                requireContext().showToast(getString(R.string.account_name_updated))
                LogCapture.i(TAG, "Display name updated to: $newName")
            } catch (e: Exception) {
                LogCapture.e(TAG, "Failed to update name: ${e.message}", e)
                requireContext().showToast("Erreur: ${e.localizedMessage}")
            }
        }
    }

    private fun exportData() {
        val user = FirebaseAuth.getInstance().currentUser ?: return

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val result = firestoreRepository.getPredictionsByUserOnce(user.uid)
                val predictions = result.getOrDefault(emptyList())

                if (predictions.isEmpty()) {
                    requireContext().showToast(getString(R.string.account_export_empty))
                    return@launch
                }

                val jsonArray = JSONArray()
                val dateFormat = SimpleDateFormat("yyyy-MM-dd'T'HH:mm:ss", Locale.FRANCE)

                predictions.forEach { p ->
                    val obj = JSONObject().apply {
                        put("age", p.predictedAge)
                        put("age_confidence", p.ageConfidence)
                        put("gender", p.predictedGender)
                        put("gender_confidence", p.genderConfidence)
                        put("ethnicity", p.predictedEthnicity)
                        put("ethnicity_confidence", p.ethnicityConfidence)
                        p.createdAt?.let {
                            put("date", dateFormat.format(it.toDate()))
                        }
                    }
                    jsonArray.put(obj)
                }

                val exportJson = JSONObject().apply {
                    put("user", user.email)
                    put("export_date", dateFormat.format(Date()))
                    put("total_predictions", predictions.size)
                    put("predictions", jsonArray)
                }

                // Share via intent
                val shareIntent = Intent(Intent.ACTION_SEND).apply {
                    type = "application/json"
                    putExtra(Intent.EXTRA_TEXT, exportJson.toString(2))
                    putExtra(Intent.EXTRA_SUBJECT, "FacePredictor - Export données")
                }
                startActivity(Intent.createChooser(shareIntent, "Exporter via..."))

                LogCapture.i(TAG, "Data exported: ${predictions.size} predictions")
            } catch (e: Exception) {
                LogCapture.e(TAG, "Failed to export data: ${e.message}", e)
                requireContext().showToast(getString(R.string.account_export_error))
            }
        }
    }

    private fun showChangePasswordDialog() {
        val dialogView = LayoutInflater.from(requireContext())
            .inflate(R.layout.dialog_change_password, null)

        val etCurrentPassword = dialogView.findViewById<TextInputEditText>(R.id.etCurrentPassword)
        val etNewPassword = dialogView.findViewById<TextInputEditText>(R.id.etNewPassword)
        val etConfirmPassword = dialogView.findViewById<TextInputEditText>(R.id.etConfirmPassword)

        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.account_change_password)
            .setView(dialogView)
            .setPositiveButton("Modifier") { _, _ ->
                val currentPassword = etCurrentPassword.text.toString()
                val newPassword = etNewPassword.text.toString()
                val confirmPassword = etConfirmPassword.text.toString()

                if (currentPassword.isEmpty() || newPassword.isEmpty() || confirmPassword.isEmpty()) {
                    requireContext().showToast(getString(R.string.error_empty_fields))
                    return@setPositiveButton
                }

                if (newPassword != confirmPassword) {
                    requireContext().showToast(getString(R.string.error_password_mismatch))
                    return@setPositiveButton
                }

                if (newPassword.length < 6) {
                    requireContext().showToast("Le mot de passe doit faire au moins 6 caractères")
                    return@setPositiveButton
                }

                changePassword(currentPassword, newPassword)
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun showSetPasswordDialog() {
        val dialogView = LayoutInflater.from(requireContext())
            .inflate(R.layout.dialog_set_password, null)

        val etNewPassword = dialogView.findViewById<TextInputEditText>(R.id.etNewPassword)
        val etConfirmPassword = dialogView.findViewById<TextInputEditText>(R.id.etConfirmPassword)

        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.account_set_password)
            .setView(dialogView)
            .setPositiveButton("Définir") { _, _ ->
                val newPassword = etNewPassword.text.toString()
                val confirmPassword = etConfirmPassword.text.toString()

                if (newPassword.isEmpty() || confirmPassword.isEmpty()) {
                    requireContext().showToast(getString(R.string.error_empty_fields))
                    return@setPositiveButton
                }

                if (newPassword != confirmPassword) {
                    requireContext().showToast(getString(R.string.error_password_mismatch))
                    return@setPositiveButton
                }

                if (newPassword.length < 6) {
                    requireContext().showToast("Le mot de passe doit faire au moins 6 caractères")
                    return@setPositiveButton
                }

                setPassword(newPassword)
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun changePassword(currentPassword: String, newPassword: String) {
        val user = FirebaseAuth.getInstance().currentUser ?: return
        val email = user.email ?: return

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val credential = EmailAuthProvider.getCredential(email, currentPassword)
                user.reauthenticate(credential).await()
                user.updatePassword(newPassword).await()
                requireContext().showToast(getString(R.string.account_password_updated))
                LogCapture.i(TAG, "Password changed successfully")
            } catch (e: Exception) {
                LogCapture.e(TAG, "Failed to change password: ${e.message}", e)
                requireContext().showToast("Erreur: ${e.localizedMessage}")
            }
        }
    }

    private fun setPassword(newPassword: String) {
        val user = FirebaseAuth.getInstance().currentUser ?: return
        val email = user.email ?: return

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val credential = EmailAuthProvider.getCredential(email, newPassword)
                user.linkWithCredential(credential).await()
                isGoogleUser = false
                setupUserInfo()
                setupSecurityInfo()
                requireContext().showToast(getString(R.string.account_password_set))
                LogCapture.i(TAG, "Password set successfully for Google user")
            } catch (e: Exception) {
                LogCapture.e(TAG, "Failed to set password: ${e.message}", e)
                requireContext().showToast("Erreur: ${e.localizedMessage}")
            }
        }
    }

    private fun showDeleteAccountConfirmation() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.account_delete)
            .setMessage(R.string.account_delete_confirm)
            .setIcon(R.drawable.ic_delete)
            .setPositiveButton("Supprimer") { _, _ ->
                if (isGoogleUser) {
                    reauthenticateWithGoogleAndDelete()
                } else {
                    showPasswordForDeletion()
                }
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun showPasswordForDeletion() {
        val dialogView = LayoutInflater.from(requireContext())
            .inflate(R.layout.dialog_delete_account, null)

        val etPassword = dialogView.findViewById<TextInputEditText>(R.id.etPassword)

        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.account_delete)
            .setView(dialogView)
            .setPositiveButton("Supprimer") { _, _ ->
                val password = etPassword.text.toString()
                if (password.isEmpty()) {
                    requireContext().showToast(getString(R.string.error_empty_fields))
                    return@setPositiveButton
                }
                deleteAccountWithPassword(password)
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun deleteAccountWithPassword(password: String) {
        val user = FirebaseAuth.getInstance().currentUser ?: return
        val email = user.email ?: return

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val credential = EmailAuthProvider.getCredential(email, password)
                user.reauthenticate(credential).await()
                firestoreRepository.clearHistory(user.uid)
                user.delete().await()
                requireContext().showToast(getString(R.string.account_delete_success))
                LogCapture.i(TAG, "Account deleted successfully")
                navigateToLogin()
            } catch (e: Exception) {
                LogCapture.e(TAG, "Failed to delete account: ${e.message}", e)
                requireContext().showToast("Erreur: ${e.localizedMessage}")
            }
        }
    }

    private fun reauthenticateWithGoogleAndDelete() {
        val credentialManager = CredentialManager.create(requireContext())

        val googleIdOption = GetGoogleIdOption.Builder()
            .setFilterByAuthorizedAccounts(true)
            .setServerClientId(WEB_CLIENT_ID)
            .build()

        val request = GetCredentialRequest.Builder()
            .addCredentialOption(googleIdOption)
            .build()

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                val result = credentialManager.getCredential(
                    request = request,
                    context = requireActivity()
                )

                val credential = result.credential
                if (credential is CustomCredential &&
                    credential.type == GoogleIdTokenCredential.TYPE_GOOGLE_ID_TOKEN_CREDENTIAL
                ) {
                    val googleCredential = GoogleIdTokenCredential.createFrom(credential.data)
                    val authCredential = GoogleAuthProvider.getCredential(googleCredential.idToken, null)

                    val user = FirebaseAuth.getInstance().currentUser ?: return@launch
                    user.reauthenticate(authCredential).await()
                    firestoreRepository.clearHistory(user.uid)
                    user.delete().await()
                    requireContext().showToast(getString(R.string.account_delete_success))
                    LogCapture.i(TAG, "Google account deleted successfully")
                    navigateToLogin()
                } else {
                    requireContext().showToast(getString(R.string.account_delete_reauth_error))
                }
            } catch (e: GetCredentialException) {
                LogCapture.e(TAG, "Google re-auth cancelled: ${e.message}", e)
                requireContext().showToast(getString(R.string.account_delete_reauth_error))
            } catch (e: Exception) {
                LogCapture.e(TAG, "Failed to delete account: ${e.message}", e)
                requireContext().showToast("Erreur: ${e.localizedMessage}")
            }
        }
    }

    private fun showLogoutConfirmation() {
        MaterialAlertDialogBuilder(requireContext())
            .setTitle(R.string.account_logout)
            .setMessage(R.string.account_logout_confirm)
            .setPositiveButton("Déconnexion") { _, _ ->
                logout()
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun logout() {
        authService.logout()
        navigateToLogin()
    }

    private fun navigateToLogin() {
        val intent = Intent(requireContext(), LoginActivity::class.java)
        intent.flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK
        startActivity(intent)
        requireActivity().finish()
    }

    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}