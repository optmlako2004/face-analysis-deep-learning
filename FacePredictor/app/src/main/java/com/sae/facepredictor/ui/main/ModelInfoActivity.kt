package com.sae.facepredictor.ui.main

import android.os.Bundle
import androidx.appcompat.app.AppCompatActivity
import com.sae.facepredictor.databinding.ActivityModelInfoBinding
import com.sae.facepredictor.utils.PredictionMode
import com.sae.facepredictor.utils.SessionManager
import org.json.JSONObject

class ModelInfoActivity : AppCompatActivity() {

    private lateinit var binding: ActivityModelInfoBinding
    private lateinit var sessionManager: SessionManager

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityModelInfoBinding.inflate(layoutInflater)
        setContentView(binding.root)

        sessionManager = SessionManager(this)

        setupUI()
        displayModelInfo()
    }

    private fun setupUI() {
        binding.toolbar.setNavigationOnClickListener {
            finish()
        }
    }

    private fun displayModelInfo() {
        when (sessionManager.predictionMode) {
            PredictionMode.HYBRID -> displayHybridInfo()
            PredictionMode.MULTITASK -> displayMultitaskInfo()
            PredictionMode.ORIENTED -> displayOrientedV2Info()
            PredictionMode.TEST_MOBILENET -> displayTestMobileNetInfo()
        }
    }

    private fun displayHybridInfo() {
        val genderInfo = loadModelInfo("gender_v2_model_info.json")
        val ageInfo = loadModelInfo("age_v2_model_info.json")
        val multitaskInfo = loadModelInfo("multitask_model_info.json")

        // Model header info
        binding.tvModelName.text = "Face Predictor Hybride"
        binding.tvModelType.text = "Combine le meilleur des 2 approches"
        binding.tvInputSize.text = "Entree: 128x128x3 pixels (RGB)"

        // Gender metrics (from V2 - best for gender)
        binding.tvGenderMetrics.text = buildString {
            appendLine("Source: Gender V2 (meilleur)")
            appendLine("Modele: gender_v2_model.tflite")
            appendLine("Type: Classification binaire + CLAHE")
            appendLine("Classes: Homme, Femme")
            if (genderInfo != null) {
                val metrics = genderInfo.optJSONObject("metrics")
                if (metrics != null) {
                    val accuracy = metrics.optDouble("accuracy", 0.0) * 100
                    val f1 = metrics.optDouble("f1", 0.0) * 100
                    appendLine("Precision: ${String.format("%.1f", accuracy)}%")
                    appendLine("F1-Score: ${String.format("%.1f", f1)}%")
                }
            }
        }

        // Age metrics (from V2)
        binding.tvAgeMetrics.text = buildString {
            appendLine("Source: Age V2")
            appendLine("Modele: age_v2_model.tflite")
            appendLine("Type: Regression")
            if (ageInfo != null) {
                val metrics = ageInfo.optJSONObject("metrics")
                if (metrics != null) {
                    val mae = metrics.optDouble("mae", 0.0)
                    val rmse = metrics.optDouble("rmse", 0.0)
                    appendLine("MAE: ${String.format("%.2f", mae)} ans")
                    appendLine("RMSE: ${String.format("%.2f", rmse)} ans")
                }
            }
        }

        // Ethnicity metrics (from V4 Multitask - best for ethnicity)
        binding.tvEthnicityMetrics.text = buildString {
            appendLine("Source: Multitache V4 (meilleur)")
            appendLine("Modele: multitask_model.tflite")
            appendLine("Type: Softmax (4 classes)")
            appendLine("Classes: Blanc, Noir, Asiatique, Indien")
            if (multitaskInfo != null) {
                val metrics = multitaskInfo.optJSONObject("metrics")
                if (metrics != null) {
                    val accuracy = metrics.optDouble("ethnicity_accuracy", 0.0) * 100
                    val f1 = metrics.optDouble("ethnicity_f1_macro", 0.0) * 100
                    appendLine("Precision: ${String.format("%.1f", accuracy)}%")
                    appendLine("F1-Score (macro): ${String.format("%.1f", f1)}%")
                }
            }
        }

        // Improvements/Techniques
        binding.tvImprovements.text = buildString {
            appendLine("- Combine les forces des 2 approches")
            appendLine("- Genre: V2 Oriente (100% correct en test)")
            appendLine("- Ethnicite: V4 Multitache (meilleur)")
            appendLine("- Age: V2 Oriente")
            appendLine("- CLAHE pour le genre")
            appendLine("- EfficientNetB0 backbone")
            appendLine("- Prediction optimale par tache")
        }
    }

    private fun displayMultitaskInfo() {
        val multitaskInfo = loadModelInfo("multitask_model_info.json")

        // Model header info
        binding.tvModelName.text = "Face Predictor Multitache V4"
        binding.tvModelType.text = "1 Modele Unifie (EfficientNetB0)"
        binding.tvInputSize.text = "Entree: 128x128x3 pixels (RGB)"

        // Gender metrics
        binding.tvGenderMetrics.text = buildString {
            appendLine("Sortie: Tete de classification binaire")
            appendLine("Type: Sigmoid (0 = Homme, 1 = Femme)")
            appendLine("Classes: Homme, Femme")
            if (multitaskInfo != null) {
                val metrics = multitaskInfo.optJSONObject("metrics")
                if (metrics != null) {
                    val accuracy = metrics.optDouble("gender_accuracy", 0.0) * 100
                    val f1 = metrics.optDouble("gender_f1", 0.0) * 100
                    appendLine("Precision: ${String.format("%.1f", accuracy)}%")
                    appendLine("F1-Score: ${String.format("%.1f", f1)}%")
                }
            }
        }

        // Age metrics
        binding.tvAgeMetrics.text = buildString {
            appendLine("Sortie: Tete de regression")
            appendLine("Type: Valeur continue (0-116 ans)")
            if (multitaskInfo != null) {
                val metrics = multitaskInfo.optJSONObject("metrics")
                if (metrics != null) {
                    val mae = metrics.optDouble("age_mae", 0.0)
                    appendLine("MAE: ${String.format("%.2f", mae)} ans")
                }
            }
        }

        // Ethnicity metrics
        binding.tvEthnicityMetrics.text = buildString {
            appendLine("Sortie: Tete de classification multi-classe")
            appendLine("Type: Softmax (4 classes)")
            appendLine("Classes: Blanc, Noir, Asiatique, Indien")
            if (multitaskInfo != null) {
                val metrics = multitaskInfo.optJSONObject("metrics")
                if (metrics != null) {
                    val accuracy = metrics.optDouble("ethnicity_accuracy", 0.0) * 100
                    val f1 = metrics.optDouble("ethnicity_f1_macro", 0.0) * 100
                    appendLine("Precision: ${String.format("%.1f", accuracy)}%")
                    appendLine("F1-Score (macro): ${String.format("%.1f", f1)}%")
                }
            }
        }

        // Improvements/Techniques
        binding.tvImprovements.text = buildString {
            appendLine("- Modele multi-tache unifie")
            appendLine("- EfficientNetB0 comme backbone partage")
            appendLine("- 3 tetes de sortie specialisees")
            appendLine("- Dataset UTKFace equilibre (4 classes)")
            appendLine("- Data augmentation avancee")
            appendLine("- Balanced sampling des classes")
            appendLine("- AdamW optimizer")
            appendLine("- Apprentissage conjoint des 3 taches")
        }
    }

    private fun displayOrientedV2Info() {
        // Load model info from JSON files
        val genderInfo = loadModelInfo("gender_v2_model_info.json")
        val ageInfo = loadModelInfo("age_v2_model_info.json")
        val ethnicityInfo = loadModelInfo("ethnicity_v2_model_info.json")

        // Model header info
        binding.tvModelName.text = "Face Predictor Oriente V2"
        binding.tvModelType.text = "3 Modeles Separes (EfficientNetB0)"
        binding.tvInputSize.text = "Entree: 128x128x3 pixels (RGB)"

        // Gender metrics
        binding.tvGenderMetrics.text = buildString {
            appendLine("Modele: gender_v2_model.tflite")
            appendLine("Type: Classification binaire")
            appendLine("Classes: Homme, Femme")
            if (genderInfo != null) {
                val metrics = genderInfo.optJSONObject("metrics")
                if (metrics != null) {
                    val accuracy = metrics.optDouble("accuracy", 0.0) * 100
                    val f1 = metrics.optDouble("f1", 0.0) * 100
                    appendLine("Precision: ${String.format("%.1f", accuracy)}%")
                    appendLine("F1-Score: ${String.format("%.1f", f1)}%")
                }
            } else {
                appendLine("Precision: En attente...")
            }
            appendLine("Preprocessing: CLAHE")
        }

        // Age metrics
        binding.tvAgeMetrics.text = buildString {
            appendLine("Modele: age_v2_model.tflite")
            appendLine("Type: Regression")
            if (ageInfo != null) {
                val metrics = ageInfo.optJSONObject("metrics")
                if (metrics != null) {
                    val mae = metrics.optDouble("mae", 0.0)
                    val rmse = metrics.optDouble("rmse", 0.0)
                    appendLine("MAE: ${String.format("%.2f", mae)} ans")
                    appendLine("RMSE: ${String.format("%.2f", rmse)} ans")
                }
            } else {
                appendLine("MAE: En attente...")
            }
        }

        // Ethnicity metrics
        binding.tvEthnicityMetrics.text = buildString {
            appendLine("Modele: ethnicity_v2_model.tflite")
            appendLine("Type: Classification multi-classe")
            appendLine("Classes: Blanc, Noir, Asiatique, Indien, Autre")
            if (ethnicityInfo != null) {
                val metrics = ethnicityInfo.optJSONObject("metrics")
                if (metrics != null) {
                    val accuracy = metrics.optDouble("accuracy_simple", 0.0) * 100
                    val f1 = metrics.optDouble("f1_macro_simple", 0.0) * 100
                    appendLine("Precision: ${String.format("%.1f", accuracy)}%")
                    appendLine("F1-Score (macro): ${String.format("%.1f", f1)}%")
                }
            } else {
                appendLine("Precision: En attente...")
            }
        }

        // Improvements/Techniques
        binding.tvImprovements.text = buildString {
            appendLine("- 3 modeles specialises independants")
            appendLine("- EfficientNetB0 comme backbone")
            appendLine("- CLAHE pour le modele de genre")
            appendLine("- Squeeze-Excitation attention (age)")
            appendLine("- CBAM attention (genre)")
            appendLine("- Dataset UTKFace equilibre")
            appendLine("- Data augmentation avancee")
            appendLine("- AdamW optimizer + ReduceLROnPlateau")
            appendLine("- Early stopping pour eviter overfitting")
            appendLine("- Test Time Augmentation (TTA)")
        }
    }

    private fun displayTestMobileNetInfo() {
        binding.tvModelName.text = "Test MobileNet V3"
        binding.tvModelType.text = "3 Modeles MobileNetV3Small (Test)"
        binding.tvInputSize.text = "Entree: 128x128x3 pixels (RGB)"

        binding.tvGenderMetrics.text = buildString {
            appendLine("Modele: gender_v3_mobilenet.tflite")
            appendLine("Type: Classification binaire (MobileNetV3)")
            appendLine("Classes: Homme, Femme")
            appendLine("Precision: ~84.6%")
        }

        binding.tvAgeMetrics.text = buildString {
            appendLine("Modele: age_v3_mobilenet.tflite")
            appendLine("Type: Regression (MobileNetV3)")
            appendLine("MAE: ~8 ans")
        }

        binding.tvEthnicityMetrics.text = buildString {
            appendLine("Modele: ethnicity_v3_mobilenet.tflite")
            appendLine("Type: Classification multi-classe (MobileNetV3)")
            appendLine("Classes: Blanc, Noir, Asiatique, Indien")
            appendLine("Precision: ~42.1%")
        }

        binding.tvImprovements.text = buildString {
            appendLine("- Test de l'architecture MobileNetV3Small")
            appendLine("- Modeles plus legers (2.6-3.4 MB)")
            appendLine("- Inference plus rapide (~0.7ms)")
            appendLine("- Performances inferieures a EfficientNet")
            appendLine("- Non retenu pour la version finale")
        }
    }

    private fun loadModelInfo(filename: String): JSONObject? {
        return try {
            val jsonString = assets.open(filename).bufferedReader().use { it.readText() }
            JSONObject(jsonString)
        } catch (e: Exception) {
            null
        }
    }
}
