package com.sae.facepredictor.data.firebase

import com.google.firebase.firestore.FirebaseFirestore
import com.google.firebase.firestore.Query
import com.sae.facepredictor.data.model.PredictionResult
import kotlinx.coroutines.channels.awaitClose
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.tasks.await

/**
 * Repository pour gérer les prédictions dans Firestore.
 * Remplace l'ancien PredictionRepository qui utilisait Room.
 */
class FirestoreRepository {

    private val firestore: FirebaseFirestore = FirebaseFirestore.getInstance()
    private val predictionsCollection = firestore.collection("predictions")

    /**
     * Sauvegarde une prédiction dans Firestore
     */
    suspend fun savePrediction(
        userId: String,
        imagePath: String,
        result: PredictionResult
    ): Result<String> {
        return try {
            val prediction = hashMapOf(
                "userId" to userId,
                "imagePath" to imagePath,
                "predictedAge" to result.age,
                "ageConfidence" to result.ageConfidence,
                "predictedGender" to result.gender.label,
                "genderConfidence" to result.genderConfidence,
                "predictedEthnicity" to result.ethnicity.label,
                "ethnicityConfidence" to result.ethnicityConfidence,
                "createdAt" to com.google.firebase.Timestamp.now()
            )

            val docRef = predictionsCollection.add(prediction).await()
            Result.success(docRef.id)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Récupère les prédictions d'un utilisateur en temps réel (Flow)
     */
    fun getPredictionsByUser(userId: String): Flow<List<FirestorePrediction>> = callbackFlow {
        val listenerRegistration = predictionsCollection
            .whereEqualTo("userId", userId)
            .orderBy("createdAt", Query.Direction.DESCENDING)
            .addSnapshotListener { snapshot, error ->
                if (error != null) {
                    // Log the error but send empty list instead of crashing
                    android.util.Log.e("FirestoreRepo", "Query error: ${error.message}", error)
                    trySend(emptyList())
                    return@addSnapshotListener
                }

                val predictions = snapshot?.documents?.mapNotNull { doc ->
                    doc.toObject(FirestorePrediction::class.java)
                } ?: emptyList()

                trySend(predictions)
            }

        awaitClose { listenerRegistration.remove() }
    }

    /**
     * Récupère les prédictions d'un utilisateur (une seule fois)
     */
    suspend fun getPredictionsByUserOnce(userId: String): Result<List<FirestorePrediction>> {
        return try {
            val snapshot = predictionsCollection
                .whereEqualTo("userId", userId)
                .orderBy("createdAt", Query.Direction.DESCENDING)
                .get()
                .await()

            val predictions = snapshot.documents.mapNotNull { doc ->
                doc.toObject(FirestorePrediction::class.java)
            }

            Result.success(predictions)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Supprime une prédiction
     */
    suspend fun deletePrediction(predictionId: String): Result<Unit> {
        return try {
            predictionsCollection.document(predictionId).delete().await()
            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Supprime toutes les prédictions d'un utilisateur
     */
    suspend fun clearHistory(userId: String): Result<Unit> {
        return try {
            val snapshot = predictionsCollection
                .whereEqualTo("userId", userId)
                .get()
                .await()

            val batch = firestore.batch()
            snapshot.documents.forEach { doc ->
                batch.delete(doc.reference)
            }
            batch.commit().await()

            Result.success(Unit)
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    /**
     * Compte le nombre de prédictions d'un utilisateur
     */
    suspend fun countPredictions(userId: String): Result<Int> {
        return try {
            val snapshot = predictionsCollection
                .whereEqualTo("userId", userId)
                .get()
                .await()

            Result.success(snapshot.size())
        } catch (e: Exception) {
            Result.failure(e)
        }
    }

    companion object {
        @Volatile
        private var INSTANCE: FirestoreRepository? = null

        fun getInstance(): FirestoreRepository {
            return INSTANCE ?: synchronized(this) {
                INSTANCE ?: FirestoreRepository().also { INSTANCE = it }
            }
        }
    }
}
