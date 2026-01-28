package com.sae.facepredictor.ui.main

import android.content.Intent
import android.os.Bundle
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.appcompat.app.AlertDialog
import androidx.fragment.app.Fragment
import androidx.lifecycle.lifecycleScope
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputEditText
import com.google.android.material.textfield.TextInputLayout
import com.google.firebase.auth.EmailAuthProvider
import com.google.firebase.auth.FirebaseAuth
import com.google.firebase.auth.GoogleAuthProvider
import com.sae.facepredictor.R
import com.sae.facepredictor.data.firebase.FirebaseAuthService
import com.sae.facepredictor.data.firebase.FirestoreRepository
import com.sae.facepredictor.databinding.FragmentAccountBinding
import com.sae.facepredictor.ui.auth.LoginActivity
import com.sae.facepredictor.utils.LogCapture
import com.sae.facepredictor.utils.showToast
import kotlinx.coroutines.launch
import kotlinx.coroutines.tasks.await

class AccountFragment : Fragment() {

    companion object {
        private const val TAG = "AccountFragment"
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
        setupListeners()
    }

    private fun setupUserInfo() {
        val user = FirebaseAuth.getInstance().currentUser ?: return

        // Name
        val displayName = user.displayName
            ?: user.email?.substringBefore("@")
            ?: "Utilisateur"
        binding.tvUserName.text = displayName

        // Email
        binding.tvUserEmail.text = user.email ?: "Email non disponible"

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

    private fun setupListeners() {
        binding.cardChangePassword.setOnClickListener {
            if (isGoogleUser) {
                showSetPasswordDialog()
            } else {
                showChangePasswordDialog()
            }
        }

        binding.cardDeleteAccount.setOnClickListener {
            showDeleteAccountConfirmation()
        }

        binding.btnLogout.setOnClickListener {
            showLogoutConfirmation()
        }
    }

    private fun showChangePasswordDialog() {
        val dialogView = LayoutInflater.from(requireContext())
            .inflate(R.layout.dialog_change_password, null)

        val tilCurrentPassword = dialogView.findViewById<TextInputLayout>(R.id.tilCurrentPassword)
        val tilNewPassword = dialogView.findViewById<TextInputLayout>(R.id.tilNewPassword)
        val tilConfirmPassword = dialogView.findViewById<TextInputLayout>(R.id.tilConfirmPassword)
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
                // Re-authenticate
                val credential = EmailAuthProvider.getCredential(email, currentPassword)
                user.reauthenticate(credential).await()

                // Update password
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
                // Link email/password credential
                val credential = EmailAuthProvider.getCredential(email, newPassword)
                user.linkWithCredential(credential).await()

                isGoogleUser = false
                setupUserInfo()

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
                deleteAccount()
            }
            .setNegativeButton("Annuler", null)
            .show()
    }

    private fun deleteAccount() {
        val user = FirebaseAuth.getInstance().currentUser ?: return
        val userId = user.uid

        viewLifecycleOwner.lifecycleScope.launch {
            try {
                // Delete user data from Firestore
                firestoreRepository.clearHistory(userId)

                // Delete Firebase Auth user
                user.delete().await()

                requireContext().showToast(getString(R.string.account_delete_success))
                LogCapture.i(TAG, "Account deleted successfully")

                // Navigate to login
                navigateToLogin()
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
