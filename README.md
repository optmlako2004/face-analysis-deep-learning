# FacePredictor - Application Mobile Intelligente

## Vue d'Ensemble du Projet

**SAE BUT 3 Informatique - Annee 2025-2026**

Ce projet developpe une **application mobile Android intelligente** capable de predire l'age, le genre et l'ethnicite d'une personne a partir de son visage, en utilisant des modeles de Deep Learning optimises bases sur **EfficientNetB0** et deployes avec TensorFlow Lite.

### Fonctionnalites de l'Application

| Fonctionnalite       | Description                                    | Statut |
| -------------------- | ---------------------------------------------- | ------ |
| Capture camera       | Prendre une photo via la camera                | Fait   |
| Import galerie       | Charger une photo existante                    | Fait   |
| Detection de visage  | MediaPipe Face Detection                       | Fait   |
| Prediction IA        | Age, Genre, Ethnicite                          | Fait   |
| Prediction temps reel| Analyse en streaming video                     | Fait   |
| Switch de modele     | 4 modes: Hybride, Oriente V2, Multitache V4, Test MobileNet | Fait   |
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

### Modeles Principaux (EfficientNetB0)

L'application utilise des modeles bases sur **EfficientNetB0** pour la production :

| Modele                          | Taille    | Description                        |
| ------------------------------- | --------- | ---------------------------------- |
| `gender_v2_model.tflite`       | 5.2 MB    | Classification binaire du genre (+ CLAHE) |
| `age_v2_model.tflite`          | 6.0 MB    | Regression de l'age               |
| `ethnicity_v2_model.tflite`    | 6.0 MB    | Classification 5 classes           |
| `multitask_model.tflite`       | 10.8 MB   | Modele unifie 3 taches (V4)       |

### Modeles Test MobileNet (non retenus)

Des modeles MobileNetV3 ont ete testes mais non retenus pour la version finale en raison de performances insuffisantes :

| Modele                          | Taille    | Description                        |
| ------------------------------- | --------- | ---------------------------------- |
| `gender_v3_mobilenet.tflite`    | 2.6 MB    | Classification binaire du genre    |
| `age_v3_mobilenet.tflite`       | 3.4 MB    | Regression de l'age                |
| `ethnicity_v3_mobilenet.tflite` | 3.4 MB    | Classification 4 classes           |

### Modes de Prediction

| Mode              | Description                                | Modeles utilises                           |
| ----------------- | ------------------------------------------ | ------------------------------------------ |
| **Hybride** (recommande) | Combine le meilleur de chaque modele + Ensemble & TTA | Gender V2 (CLAHE) + Age Ensemble V2+V4 (TTA) + Ethnicity V4 (TTA) |
| **Oriente V2**    | 3 modeles specialises independants         | Gender V2, Age V2, Ethnicity V2            |
| **Multitache V4** | 1 modele unifie pour les 3 taches          | multitask_model (EfficientNet V4)          |
| **Test MobileNet**| Test de performance MobileNetV3            | Gender V3, Age V3, Ethnicity V3 MobileNet  |

### Technique Ensemble + TTA (Mode Hybride)

Le mode Hybride utilise deux techniques d'inference avancees pour ameliorer la precision **sans re-entrainement** :

**Ensemble** : Moyenne des predictions de deux modeles (V2 specialise + V4 multitache) pour reduire les erreurs individuelles.

**TTA (Test Time Augmentation)** : Chaque image est predite 2 fois (originale + flip horizontal), puis les resultats sont moyennes. Cela reduit la sensibilite a l'orientation du visage.

Pour l'age, 4 predictions sont moyennees :
```
Age final = (Age_V2 + Age_V2_flip + Age_V4 + Age_V4_flip) / 4
```

Pour l'ethnicite, les softmax de V4 sont moyennes :
```
Ethnicity = argmax( (probs_V4 + probs_V4_flip) / 2 )
```

**Resultats** : MAE Age reduit de 7.57 a **6.71 ans** (-11.4%) sans aucun re-entrainement.

### Performances des Modeles

#### Comparaison des 3 Modes EfficientNet (200 images UTKFace)

| Metrique          | Hybride (Ensemble+TTA) | Oriente V2   | Multitache V4 |
| ----------------- | ---------------------- | ------------ | ------------- |
| Genre Accuracy    | **91.5%**              | 91.5%        | 81.0%         |
| Age MAE           | **6.7 ans**            | 7.6 ans      | 7.2 ans       |
| Ethnicite Acc     | **62.5%**              | 30.5%        | 62.5%         |

#### Comparaison EfficientNet vs MobileNet (Multitask)

| Metrique          | EfficientNet V4 | MobileNet V5 | Difference       |
| ----------------- | --------------- | ------------ | ---------------- |
| Age MAE           | 6.65 ans        | 7.87 ans     | +1.22 ans        |
| Genre Accuracy    | 78.1%           | 77.9%        | -0.2%            |
| Ethnicite Acc     | 64.3%           | 59.2%        | -5.1%            |
| **Taille TFLite** | 10.77 MB        | 3.51 MB      | -67%             |
| **Temps inference**| 5.33 ms        | 0.70 ms      | 7.6x plus rapide |

**Conclusion** : MobileNetV3 est plus leger et rapide mais les performances de prediction sont significativement inferieures, notamment pour l'ethnicite (-5.1%). **EfficientNetB0 a ete retenu** pour la version finale.

#### Modeles Separes V2 EfficientNet

| Modele         | Metrique   | Valeur    |
| -------------- | ---------- | --------- |
| Gender V2      | Accuracy   | ~90%      |
| Age V2         | MAE        | ~6 ans    |
| Ethnicity V2   | Accuracy   | ~70%      |

**Recommandation** : Le mode **Hybride** est recommande pour le meilleur equilibre precision/performance grace a l'Ensemble + TTA.

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
4. Creer l'index composite Firestore : collection `predictions`, champs `userId` (ASC) + `createdAt` (DESC)
5. Telecharger `google-services.json` dans `FacePredictor/app/`

---

## Notebooks d'Entrainement

### Modeles MobileNetV3 (Test)

| Notebook                              | Description                     |
| ------------------------------------- | ------------------------------- |
| `age_model_v3_mobilenet.ipynb`        | Age avec CBAM attention         |
| `gender_model_v3_mobilenet.ipynb`     | Genre avec CLAHE preprocessing  |
| `ethnicity_model_v3_mobilenet.ipynb`  | Ethnicite 4 classes             |
| `multitask_mobilenet.ipynb`           | Multi-tache unifie              |

### Comparaisons et Visualisations

| Notebook                                    | Description                          |
| ------------------------------------------- | ------------------------------------ |
| `mobilenet_vs_efficientnet_comparison.ipynb`| Benchmark EfficientNet vs MobileNet  |
| `grayscale_vs_rgb_experiment.ipynb`         | Impact de la couleur                 |
| `age_intervals_and_filter_visualization.ipynb` | Grad-CAM et feature maps          |

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

**Q: Pourquoi EfficientNetB0 au lieu de MobileNetV3 ?**
R: MobileNetV3 a ete teste mais les performances de prediction sont significativement inferieures (-5% ethnicite, +1.2 ans MAE age). EfficientNetB0 offre un meilleur equilibre precision/taille pour le mobile.

**Q: Quel mode de prediction choisir ?**
R: Le mode **Hybride** est recommande (meilleur equilibre). Il combine le genre V2 (CLAHE), l'age par Ensemble V2+V4 avec TTA, et l'ethnicite du multitache V4 avec TTA.

**Q: Qu'est-ce que l'Ensemble + TTA ?**
R: L'Ensemble moyenne les predictions de deux modeles differents (V2 + V4) pour reduire les erreurs. Le TTA (Test Time Augmentation) predit sur l'image originale et son miroir horizontal, puis moyenne les resultats. Cela ameliore la MAE age de 7.57 a 6.71 ans sans re-entrainement.

**Q: Pourquoi garder le mode Test MobileNet ?**
R: Pour montrer la demarche de test et comparaison des architectures. Les modeles MobileNetV3 sont accessibles dans les parametres a titre experimental.

**Q: Les predictions sont incorrectes sur mes photos**
R: Verifiez que le visage est bien detecte et recadre. Les photos de groupe ou corps entier ne fonctionnent pas.

**Q: Comment ameliorer la precision ?**
R: Utilisez des photos de bonne qualite, visage de face, eclairage uniforme.

---

## Livrables du Projet

| Livrable                                    | Statut    |
| ------------------------------------------- | --------- |
| Application Android complete                | Fait      |
| Modeles EfficientNetB0 (4 fichiers TFLite)  | Fait      |
| Ensemble + TTA (amelioration age -11.4%)    | Fait      |
| Test MobileNetV3 (comparaison)              | Fait      |
| Detection visage (MediaPipe)                | Fait      |
| Guide visuel temps reel (ovale)             | Fait      |
| Authentification Firebase                   | Fait      |
| Prediction temps reel                       | Fait      |
| Historique des predictions (Firestore)      | Fait      |
| 4 modes de prediction                       | Fait      |
| Gestion compte RGPD                         | Fait      |
| Comparaison MobileNet vs EfficientNet       | Fait      |
| Notebooks d'entrainement                    | Fait      |
| Documentation                               | Fait      |

---

_Projet SAE BUT3 Informatique - Fevrier 2026_
_Encadrement: Bilal Faye & Hanane Azzag (LIPN, CNRS UMR 7030, Universite Sorbonne Paris Nord)_
