# FacePredictor - Application Mobile Intelligente

## Vue d'Ensemble du Projet

**SAE BUT 3 Informatique - Annee 2025-2026**

Ce projet developpe une **application mobile Android intelligente** capable de predire l'age, le genre et l'ethnicite d'une personne a partir de son visage, en utilisant des modeles de Deep Learning optimises bases sur **MobileNetV3** et deployes avec TensorFlow Lite.

### Fonctionnalites de l'Application

| Fonctionnalite       | Description                                    | Statut |
| -------------------- | ---------------------------------------------- | ------ |
| Capture camera       | Prendre une photo via la camera                | Fait   |
| Import galerie       | Charger une photo existante                    | Fait   |
| Detection de visage  | MediaPipe Face Detection                       | Fait   |
| Prediction IA        | Age, Genre, Ethnicite                          | Fait   |
| Prediction temps reel| Analyse en streaming video                     | Fait   |
| Switch de modele     | 3 modes: Hybride, MobileNet V3, Multitask V5   | Fait   |
| Authentification     | Firebase Auth (Email + Google Sign-In)         | Fait   |
| Historique           | Sauvegarder les predictions (Firestore)        | Fait   |
| Gestion compte       | Mot de passe, suppression RGPD                 | Fait   |

---

## Architecture de l'Application

L'application est structuree en **4 onglets** :

| Onglet        | Description                                              |
| ------------- | -------------------------------------------------------- |
| **Accueil**   | Presentation de l'app, statistiques, video tutoriel      |
| **Prediction**| Camera, Temps reel, Galerie, Historique                  |
| **Parametres**| Selection du mode de prediction, logs de debug           |
| **Compte**    | Profil, mot de passe, suppression compte (RGPD)          |

---

## Modeles de Prediction

### Modeles Utilises (MobileNetV3)

L'application utilise exclusivement des modeles bases sur **MobileNetV3** pour un deploiement mobile optimal :

| Modele                          | Taille    | Description                        |
| ------------------------------- | --------- | ---------------------------------- |
| `gender_v3_mobilenet.tflite`    | 2.6 MB    | Classification binaire du genre    |
| `age_v3_mobilenet.tflite`       | 3.4 MB    | Regression de l'age                |
| `ethnicity_v3_mobilenet.tflite` | 3.4 MB    | Classification 4 classes           |
| `multitask_model_v5_mobilenet.tflite` | 3.5 MB | Modele unifie 3 taches        |

**Total : ~13 MB** (vs ~29 MB avec EfficientNet)

### Modes de Prediction

| Mode              | Description                                | Modeles utilises                           |
| ----------------- | ------------------------------------------ | ------------------------------------------ |
| **Hybride**       | Combine le meilleur de chaque modele       | Gender V3 + Age V3 + Multitask V5 (eth)    |
| **MobileNet V3**  | 3 modeles specialises independants         | Gender V3, Age V3, Ethnicity V3            |
| **Multitask V5**  | 1 modele unifie pour les 3 taches          | multitask_model_v5_mobilenet               |

### Performances des Modeles

#### Comparaison MobileNet vs EfficientNet (Multitask)

| Metrique          | EfficientNet V4 | MobileNet V5 | Difference       |
| ----------------- | --------------- | ------------ | ---------------- |
| Age MAE           | 6.65 ans        | 7.87 ans     | +1.22 ans        |
| Genre Accuracy    | 78.1%           | 77.9%        | -0.2%            |
| Ethnicite Acc     | 64.3%           | 59.2%        | -5.1%            |
| **Taille TFLite** | 10.77 MB        | 3.51 MB      | **-67%**         |
| **Temps inference**| 5.33 ms        | 0.70 ms      | **7.6x plus rapide** |

#### Modeles Separes V3 MobileNet

| Modele         | Metrique   | Valeur    |
| -------------- | ---------- | --------- |
| Gender V3      | Accuracy   | 84.6%     |
| Gender V3      | F1-Score   | 0.848     |
| Age V3         | MAE        | 8.04 ans  |
| Ethnicity V3   | Accuracy   | 42.1%     |

**Recommandation** : Le mode **Hybride** est recommande pour le meilleur equilibre precision/performance.

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
|-- Notebooks MobileNetV3
|   |-- age_model_v3_mobilenet.ipynb          # Age specialise
|   |-- gender_model_v3_mobilenet.ipynb       # Genre specialise
|   |-- ethnicity_model_v3_mobilenet.ipynb    # Ethnicite specialise
|   |-- multitask_model_v5_mobilenet.ipynb    # Multi-tache
|
|-- Notebooks Experimentation
|   |-- mobilenet_vs_efficientnet_comparison.ipynb  # Comparaison backbones
|   |-- grayscale_vs_rgb_experiment.ipynb           # RGB vs Grayscale
|   |-- age_intervals_and_filter_visualization.ipynb # Visualisation CNN
|
|-- artifacts/                                # Modeles et metriques
|   |-- age_v3_mobilenet.tflite
|   |-- age_v3_mobilenet_info.json
|   |-- gender_v3_mobilenet.tflite
|   |-- gender_v3_mobilenet_info.json
|   |-- ethnicity_v3_mobilenet.tflite
|   |-- ethnicity_v3_mobilenet_info.json
|   |-- multitask_model_v5_mobilenet.tflite
|   |-- multitask_v5_mobilenet_info.json
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

| Technologie       | Version | Usage                           |
| ----------------- | ------- | ------------------------------- |
| Kotlin            | 1.9.20  | Langage principal               |
| Android SDK       | 34      | Target SDK                      |
| TensorFlow Lite   | 2.17.0  | Inference des modeles           |
| MediaPipe         | 0.10.9  | Detection de visage             |
| Firebase Auth     | BoM 33  | Authentification                |
| Firestore         | BoM 33  | Stockage des predictions        |
| CameraX           | 1.3.1   | Capture photo et video          |
| Navigation        | 2.7.6   | Navigation entre fragments      |

### Installation

1. Ouvrir `FacePredictor/` dans Android Studio
2. Synchroniser Gradle
3. Configurer Firebase (`google-services.json`)
4. Build > Run

### Configuration Firebase

1. Creer un projet Firebase Console
2. Activer Authentication (Email + Google)
3. Activer Firestore Database
4. Telecharger `google-services.json` dans `FacePredictor/app/`

---

## Notebooks d'Entrainement

### Modeles MobileNetV3 (Production)

| Notebook                              | Description                     |
| ------------------------------------- | ------------------------------- |
| `age_model_v3_mobilenet.ipynb`        | Age avec CBAM attention         |
| `gender_model_v3_mobilenet.ipynb`     | Genre avec CLAHE preprocessing  |
| `ethnicity_model_v3_mobilenet.ipynb`  | Ethnicite 4 classes             |
| `multitask_model_v5_mobilenet.ipynb`  | Multi-tache unifie              |

### Comparaisons et Visualisations

| Notebook                                    | Description                          |
| ------------------------------------------- | ------------------------------------ |
| `mobilenet_vs_efficientnet_comparison.ipynb`| Benchmark des backbones              |
| `grayscale_vs_rgb_experiment.ipynb`         | Impact de la couleur                 |
| `age_intervals_and_filter_visualization.ipynb` | Grad-CAM et feature maps          |

---

## Detection de Visage

**IMPORTANT** : Les modeles ont ete entraines sur UTKFace (visages recadres). La detection de visage est **obligatoire** avant prediction !

L'application utilise **MediaPipe Face Detection** (~15-30ms sur mobile).

```
SANS detection :                    AVEC detection :
+----------------+                  +----------------+
|                |                  |    +------+    |
|      Person    |  --> Erreur     |    | Face |    |  --> OK
|     /|\        |                  |    +------+    |
|     / \        |                  |                |
+----------------+                  +----------------+
```

---

## FAQ

**Q: Pourquoi MobileNetV3 au lieu d'EfficientNet ?**
R: MobileNetV3 est 3x plus leger et 7x plus rapide avec une perte de precision marginale (-2-5%). Ideal pour le mobile.

**Q: Quel mode de prediction choisir ?**
R: Le mode **Hybride** est recommande (meilleur equilibre). Multitask V5 pour la vitesse maximale.

**Q: Les predictions sont incorrectes sur mes photos**
R: Verifiez que le visage est bien detecte et recadre. Les photos de groupe ou corps entier ne fonctionnent pas.

**Q: Comment ameliorer la precision ?**
R: Utilisez des photos de bonne qualite, visage de face, eclairage uniforme.

---

## Livrables du Projet

| Livrable                                    | Statut    |
| ------------------------------------------- | --------- |
| Application Android complete                | Fait      |
| Modeles MobileNetV3 (4 fichiers TFLite)     | Fait      |
| Detection visage (MediaPipe)                | Fait      |
| Authentification Firebase                   | Fait      |
| Prediction temps reel                       | Fait      |
| Historique des predictions                  | Fait      |
| 3 modes de prediction                       | Fait      |
| Gestion compte RGPD                         | Fait      |
| Comparaison MobileNet vs EfficientNet       | Fait      |
| Notebooks d'entrainement                    | Fait      |
| Documentation                               | Fait      |

---

_Projet SAE BUT3 Informatique - Janvier 2026_
_Encadrement: Bilal Faye & Hanane Azzag (LIPN, CNRS UMR 7030, Universite Sorbonne Paris Nord)_
