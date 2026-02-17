# FacePredictor - Application Mobile Intelligente

## Vue d'Ensemble du Projet

**SAE BUT 3 Informatique - Annee 2025-2026**

Ce projet developpe une **application mobile Android intelligente** capable de predire l'age, le genre et l'ethnicite d'une personne a partir de son visage, en utilisant des modeles de Deep Learning deployes avec TensorFlow Lite. Le modele recommande est une architecture **Mixture of Experts (MoE)** avec backbone MobileNetV3, qui surpasse les modeles EfficientNetB0 sur toutes les metriques tout en etant 29x plus rapide.

### Fonctionnalites de l'Application

| Fonctionnalite        | Description                                                     | Statut |
| --------------------- | --------------------------------------------------------------- | ------ |
| Capture camera        | Prendre une photo via la camera                                 | Fait   |
| Import galerie        | Charger une photo existante                                     | Fait   |
| Detection de visage   | MediaPipe Face Detection                                        | Fait   |
| Prediction IA         | Age, Genre, Ethnicite                                           | Fait   |
| Prediction temps reel | Analyse en streaming video                                      | Fait   |
| Switch de modele      | 4 modes: MoE Expert, Hybride, Oriente V2, Multitache V4        | Fait   |
| Authentification      | Firebase Auth (Email + Google Sign-In)                          | Fait   |
| Historique            | Sauvegarder les predictions (Firestore)                         | Fait   |
| Gestion compte        | Mot de passe, suppression RGPD                                  | Fait   |

---

## Architecture de l'Application

L'application est structuree en **4 onglets** :

| Onglet         | Description                                         |
| -------------- | --------------------------------------------------- |
| **Accueil**    | Presentation de l'app, statistiques, video tutoriel |
| **Prediction** | Camera, Temps reel, Galerie, Historique             |
| **Parametres** | Selection du mode de prediction, logs de debug      |
| **Compte**     | Profil, mot de passe, suppression compte (RGPD)     |

---

## Modeles de Prediction

### Modele Recommande : Mixture of Experts (MoE)

| Modele                   | Taille | Description                       |
| ------------------------ | ------ | --------------------------------- |
| `moe_mobilenetv3.tflite` | 6.5 MB | MoE multi-tache (3 experts/tache) |

Architecture MoE avec backbone **MobileNetV3Small** partage et 3 tetes specialisees (Genre, Age, Ethnicite), chacune composee de 3 experts MLP avec un gating network adaptatif. Entrainement en 2 phases :

1. **Warmup** (25 epochs) : backbone gele, seules les tetes MoE apprennent
2. **Fine-tuning progressif** (50 epochs) : defreeze progressif (30 → 60 → toutes les couches) avec BatchNorm gele et LR tres faible (2e-5)

Le preprocessing Rescaling `[-1, 1]` est integre directement dans le modele pour garantir la compatibilite avec les poids ImageNet de MobileNetV3.

### Modeles EfficientNetB0 (modes alternatifs)

| Modele                      | Taille  | Description                               |
| --------------------------- | ------- | ----------------------------------------- |
| `gender_v2_model.tflite`    | 5.2 MB  | Classification binaire du genre (+ CLAHE) |
| `age_v2_model.tflite`       | 6.0 MB  | Regression de l'age                       |
| `ethnicity_v2_model.tflite` | 6.0 MB  | Classification 5 classes                  |
| `multitask_model.tflite`    | 10.8 MB | Modele unifie 3 taches (V4)               |

### Modes de Prediction

| Mode                          | Description                                           | Modeles utilises                                                   |
| ----------------------------- | ----------------------------------------------------- | ------------------------------------------------------------------ |
| **MoE Expert** (recommande)   | Mixture of Experts - meilleur sur toutes les metriques | moe_mobilenetv3 (MobileNetV3 + 9 experts)                         |
| **Hybride**                   | Combine le meilleur de chaque modele + Ensemble & TTA | Gender V2 (CLAHE) + Age Ensemble V2+V4 (TTA) + Ethnicity V4 (TTA) |
| **Oriente V2**                | 3 modeles specialises independants                    | Gender V2, Age V2, Ethnicity V2                                    |
| **Multitache V4**             | 1 modele unifie pour les 3 taches                     | multitask_model (EfficientNet V4)                                  |

### Performances Comparatives (80 images UTKFace equilibrees)

| Mode              | Genre    | Ethnicite | Age MAE    | Temps/image |
| ----------------- | -------- | --------- | ---------- | ----------- |
| **MoE Expert**    | **91.2%** | **86.2%** | **5.58 ans** | **1.79 ms** |
| Hybride           | 83.8%    | 63.7%     | 6.40 ans   | 51.77 ms    |
| Oriente V2        | 83.8%    | 37.5%     | 7.12 ans   | 29.30 ms    |
| Multitache V4     | 76.2%    | 66.2%     | 6.55 ans   | 11.42 ms    |

### Detail MoE par Categorie

| Genre  | Accuracy | Ethnicite  | Accuracy |
| ------ | -------- | ---------- | -------- |
| Homme  | 95.0%    | Blanc      | 95.0%    |
| Femme  | 87.5%    | Noir       | 95.0%    |
|        |          | Asiatique  | 75.0%    |
|        |          | Indien     | 80.0%    |

**Conclusion** : Le MoE est le meilleur modele sur **toutes les metriques** simultanement (genre, ethnicite, age, vitesse). Il est **29x plus rapide** que le mode Hybride tout en etant plus precis.

### Technique Ensemble + TTA (Mode Hybride)

Le mode Hybride utilise deux techniques d'inference avancees :

**Ensemble** : Moyenne des predictions de deux modeles (V2 specialise + V4 multitache) pour reduire les erreurs individuelles.

**TTA (Test Time Augmentation)** : Chaque image est predite 2 fois (originale + flip horizontal), puis les resultats sont moyennes.

Pour l'age, 4 predictions sont moyennees :

```
Age final = (Age_V2 + Age_V2_flip + Age_V4 + Age_V4_flip) / 4
```

Pour l'ethnicite, les softmax de V4 sont moyennes :

```
Ethnicity = argmax( (probs_V4 + probs_V4_flip) / 2
```

---

## Demarrage Rapide

### Prerequis

- Python 3.10+ (recommande: 3.11 ou 3.12)
- pip (gestionnaire de paquets Python)
- Git
- 8 Go de RAM minimum (16 Go recommande)
- GPU NVIDIA avec CUDA 12.x (optionnel)

### Installation

```bash
# Cloner le projet
git clone https://github.com/VOTRE_USERNAME/SAE.git
cd SAE

# Creer et activer l'environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# Installer les dependances
pip install --upgrade pip
pip install -r requirements.txt

# Verifier l'installation
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"
```

### Telecharger le Dataset UTKFace

```bash
mkdir -p data
pip install kaggle
kaggle datasets download -d jangedoo/utkface-new
unzip utkface-new.zip -d data/
```

---

## Structure du Projet

```
SAE/
|-- Notebooks EfficientNet (Production)
|   |-- (entrainement des modeles V2 et V4)
|
|-- Notebooks MobileNetV3 (Test)
|   |-- age_model_v3_mobilenet.ipynb          # Age MobileNet (test)
|   |-- gender_model_v3_mobilenet.ipynb       # Genre MobileNet (test)
|   |-- ethnicity_model_v3_mobilenet.ipynb    # Ethnicite MobileNet (test)
|   |-- multitask_mobilenet.ipynb             # Multi-tache MobileNet (test)
|
|-- Notebooks Experimentation
|   |-- mobilenet_vs_efficientnet_comparison.ipynb  # Comparaison backbones
|   |-- grayscale_vs_rgb_experiment.ipynb           # RGB vs Grayscale
|   |-- age_intervals_and_filter_visualization.ipynb # Visualisation CNN
|
|-- training/                                 # Entrainement MoE (recommande)
|   |-- train_moe.ipynb                       # MoE MobileNetV3 (modele final)
|   |-- output_moe/                           # Artefacts MoE
|       |-- moe_mobilenetv3.keras             # Modele Keras final
|       |-- moe_mobilenetv3.tflite            # Modele TFLite (6.5 MB)
|       |-- moe_best.keras                    # Meilleur checkpoint Phase 2
|       |-- moe_warmup_best.keras             # Meilleur checkpoint Warmup
|       |-- phase1_curves.png                 # Courbes warmup
|       |-- phase2_curves.png                 # Courbes fine-tuning progressif
|       |-- confusion_matrices.png            # Matrices de confusion
|       |-- age_analysis.png                  # Analyse erreur age
|       |-- dataset_distribution.png          # Distribution du dataset
|
|-- artifacts/                                # Modeles et metriques
|
|-- data/UTKFace/                             # Dataset (non versionne)
|
|-- FacePredictor/                            # Application Android
|   |-- app/src/main/
|   |   |-- java/com/sae/facepredictor/
|   |   |   |-- ml/                           # Predicteurs TFLite
|   |   |   |-- ui/                           # Fragments et Activities
|   |   |   |-- data/                         # Firebase Repository
|   |   |-- assets/                           # Modeles TFLite
|   |   |-- res/                              # Layouts, strings
|
|-- README.md
|-- requirements.txt
```

---

## Application Android

### Technologies

| Technologie     | Version | Usage                      |
| --------------- | ------- | -------------------------- |
| Kotlin          | 1.9.20  | Langage principal          |
| Android SDK     | 34      | Target SDK                 |
| TensorFlow Lite | 2.17.0  | Inference des modeles      |
| MediaPipe       | 0.10.9  | Detection de visage        |
| Firebase Auth   | BoM 33  | Authentification           |
| Firestore       | BoM 33  | Stockage des predictions   |
| CameraX         | 1.3.1   | Capture photo et video     |
| Navigation      | 2.7.6   | Navigation entre fragments |

### Installation

1. Ouvrir `FacePredictor/` dans Android Studio
2. Synchroniser Gradle
3. Configurer Firebase (`google-services.json`)
4. Build > Run

### Configuration Firebase

1. Creer un projet Firebase Console
2. Activer Authentication (Email + Google)
3. Activer Firestore Database
4. Creer l'index composite Firestore : collection `predictions`, champs `userId` (ASC) + `createdAt` (DESC)
5. Telecharger `google-services.json` dans `FacePredictor/app/`

---

## Notebooks d'Entrainement

### Modeles MobileNetV3 (Test)

| Notebook                             | Description                    |
| ------------------------------------ | ------------------------------ |
| `age_model_v3_mobilenet.ipynb`       | Age avec CBAM attention        |
| `gender_model_v3_mobilenet.ipynb`    | Genre avec CLAHE preprocessing |
| `ethnicity_model_v3_mobilenet.ipynb` | Ethnicite 4 classes            |
| `multitask_mobilenet.ipynb`          | Multi-tache unifie             |

### Mixture of Experts (MoE) - Modele Recommande

| Notebook                   | Description                               |
| -------------------------- | ----------------------------------------- |
| `training/train_moe.ipynb` | MoE multi-tache avec MobileNetV3 backbone |

Architecture MoE : backbone partage (MobileNetV3Small pretrained ImageNet) avec Rescaling integre + 3 tetes MoE (3 experts + gating network chacune). Entrainement en 2 phases :

1. **Warmup** (25 epochs) : backbone entierement gele, seules les tetes MoE apprennent
2. **Fine-tuning progressif** (50 epochs) : defreeze progressif par paliers (30 → 60 → toutes les couches) avec BatchNorm gele et LR=2e-5

### Comparaisons et Visualisations

| Notebook                                       | Description                         |
| ---------------------------------------------- | ----------------------------------- |
| `mobilenet_vs_efficientnet_comparison.ipynb`   | Benchmark EfficientNet vs MobileNet |
| `grayscale_vs_rgb_experiment.ipynb`            | Impact de la couleur                |
| `age_intervals_and_filter_visualization.ipynb` | Grad-CAM et feature maps            |

---

## Detection de Visage

**IMPORTANT** : Les modeles ont ete entraines sur UTKFace (visages recadres). La detection de visage est **obligatoire** avant prediction !

L'application utilise **MediaPipe Face Detection** (~15-30ms sur mobile).

En mode temps reel, un **guide ovale** est affiche pour aider l'utilisateur a positionner son visage. Lorsqu'un visage est detecte, le guide disparait et un rectangle vert entoure le visage.

```
Pas de visage detecte :            Visage detecte :
+----------------+                  +----------------+
|  ############  |                  |                |
|  ##  +----+  ##|                  |   +--------+  |
|  ##  |    |  ##|  Guide ovale    |   | Visage |  |  Rectangle vert
|  ##  +----+  ##|                  |   +--------+  |
|  ############  |                  |                |
+----------------+                  +----------------+
```

---

## FAQ

**Q: Quel mode de prediction choisir ?**
R: Le mode **MoE Expert** est recommande. Il surpasse tous les autres modes sur toutes les metriques (91.2% genre, 86.2% ethnicite, 5.58 MAE age) tout en etant 29x plus rapide que le mode Hybride.

**Q: Qu'est-ce que le Mixture of Experts (MoE) ?**
R: Une architecture ou chaque tache (age, genre, ethnicite) dispose de 3 experts (petits MLP) et d'un gating network qui decide dynamiquement quel expert activer pour chaque input. Le backbone MobileNetV3Small est partage entre les 3 taches avec une couche Rescaling integree pour normaliser les pixels en [-1, 1].

**Q: Pourquoi le MoE est-il meilleur que les modeles EfficientNet ?**
R: Le MoE beneficie de la specialisation par experts : chaque expert se concentre sur un sous-ensemble de visages, et le gating network apprend a router chaque image vers l'expert le plus adapte. L'entrainement en 2 phases (warmup + fine-tuning progressif avec BatchNorm gele) evite le catastrophic forgetting.

**Q: Qu'est-ce que l'Ensemble + TTA (mode Hybride) ?**
R: L'Ensemble moyenne les predictions de deux modeles differents (V2 + V4) pour reduire les erreurs. Le TTA (Test Time Augmentation) predit sur l'image originale et son miroir horizontal, puis moyenne les resultats.

**Q: Les predictions sont incorrectes sur mes photos**
R: Verifiez que le visage est bien detecte et recadre. Les photos de groupe ou corps entier ne fonctionnent pas.

**Q: Comment ameliorer la precision ?**
R: Utilisez des photos de bonne qualite, visage de face, eclairage uniforme.

---

## Livrables du Projet

| Livrable                                        | Statut |
| ----------------------------------------------- | ------ |
| Application Android complete                    | Fait   |
| Modele MoE MobileNetV3 (recommande)             | Fait   |
| Modeles EfficientNetB0 (4 fichiers TFLite)      | Fait   |
| Ensemble + TTA (mode Hybride)                   | Fait   |
| Detection visage (MediaPipe)                     | Fait   |
| Guide visuel temps reel (ovale)                  | Fait   |
| Authentification Firebase                        | Fait   |
| Prediction temps reel                            | Fait   |
| Historique des predictions (Firestore)           | Fait   |
| 4 modes de prediction (MoE, Hybride, V2, V4)    | Fait   |
| Gestion compte RGPD                              | Fait   |
| Comparaison MobileNet vs EfficientNet            | Fait   |
| Architecture Mixture of Experts (MoE)            | Fait   |
| Notebooks d'entrainement                         | Fait   |
| Documentation                                    | Fait   |

---

_Projet SAE BUT3 Informatique - Fevrier 2026_
_Encadrement: Bilal Faye & Hanane Azzag (LIPN, CNRS UMR 7030, Universite Sorbonne Paris Nord)_
