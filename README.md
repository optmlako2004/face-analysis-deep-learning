# FacePredictor — Prediction d'age, de genre et d'ethnicite par Deep Learning

**SAE BUT 3 Informatique — Annee 2025-2026**
Universite Sorbonne Paris Nord

Application mobile Android intelligente capable de predire l'age, le genre et l'ethnicite d'une personne a partir de son visage, a l'aide de modeles de Deep Learning optimises et deployes avec TensorFlow Lite.

## Equipe

| Membre | Role |
|---|---|
| Ako Christian | Developpement ML / Android |
| Calrd Similien | Developpement Android / Integration |
| Noe Cervera | Entrainement modeles / Evaluation |
| Dhanoush Kessavane | Front-end / Design / Documentation |

**Encadrement** : Bilal Faye (LIPN, CNRS UMR 7030)

---

## Table des matieres

1. [Objectif du projet](#objectif-du-projet)
2. [Choix techniques](#choix-techniques)
3. [Les 3 strategies de modelisation](#les-3-strategies-de-modelisation)
4. [Comparaison des backbones](#comparaison-des-backbones)
5. [Donnees et preprocessing](#donnees-et-preprocessing)
6. [Application Android](#application-android)
7. [Installation et utilisation](#installation-et-utilisation)
8. [Organisation du depot](#organisation-du-depot)
9. [Metriques et evaluation](#metriques-et-evaluation)
10. [Livrables](#livrables)

---

## Objectif du projet

Le sujet demande de :

- Creer une application Android capable de capturer une image (camera), charger une photo (galerie), et effectuer des predictions en temps reel (bonus)
- Utiliser un modele pre-entraine pour la detection du visage
- Implementer **trois strategies de modelisation** distinctes
- Evaluer la performance avec les metriques : Accuracy, AUC, AP, MAE, MSE, R², ARI, NMI
- Convertir et integrer le meilleur modele dans l'application via TensorFlow Lite

---

## Choix techniques

### Cote Machine Learning (Python)

| Technologie | Usage |
|---|---|
| TensorFlow / Keras 2.20 | Entrainement des modeles |
| TensorFlow Lite | Conversion et deploiement mobile |
| UTKFace dataset | 23 708 images de visages annotes (age, genre, ethnicite) |
| Scikit-learn | Metriques d'evaluation |
| Matplotlib / Seaborn | Visualisations |

### Cote Android (Kotlin)

| Technologie | Version | Usage |
|---|---|---|
| Kotlin | 1.9.20 | Langage principal |
| Android SDK | 34 (min 24) | Compatibilite Android 7.0+ |
| TensorFlow Lite | 2.17.0 | Inference des modeles |
| MediaPipe | 0.10.9 | Detection de visage |
| Firebase Auth | BoM 33.7.0 | Authentification (Email + Google) |
| Firestore | BoM 33.7.0 | Stockage des predictions (historique) |
| CameraX | 1.3.1 | Capture photo et mode temps reel |

### Pourquoi ces choix ?

- **EfficientNetB0** comme backbone principal : meilleur compromis precision/taille apres comparaison de 4 architectures (voir section comparaison)
- **Mixture of Experts (MoE)** : architecture avancee ou chaque tache dispose de 4 experts specialises + un gating network qui route chaque image vers l'expert le plus adapte
- **MediaPipe** pour la detection de visage : rapide (~15-30ms) et fiable sur mobile
- **Firebase** pour l'authentification : integration native Android, support Email + Google Sign-In
- **Mixed precision (float16)** : reduit la memoire GPU et accelere l'entrainement

---

## Les 3 strategies de modelisation

Le sujet demande d'implementer trois approches distinctes. Voici comment nous les avons realisees :

### Strategie 1 — Trois modeles specialises

Un modele dedie par tache, chacun optimise independamment :

| Modele | Architecture | Sortie |
|---|---|---|
| Age | EfficientNetB0 + Dense | Regression lineaire (0-116 ans) |
| Genre | EfficientNetB0 + Dense | Sigmoid (Homme/Femme) |
| Ethnicite | EfficientNetB0 + Dense + Focal Loss | Softmax (5 classes) |

Chaque modele est entraine en 2 phases :
1. **Warmup** : backbone gele, seules les couches Dense apprennent
2. **Fine-tuning** : backbone debloque progressivement (BatchNorm gele)

Notebooks : `training/3_modeles_individuels.ipynb`

### Strategie 2 — Modele multi-tache (MoE)

Un seul modele predit les 3 taches simultanement grace a une architecture Mixture of Experts :

```
Image (224x224) -> EfficientNetB0 (backbone partage)
                      |
                GlobalAvgPool + BatchNorm + Dropout
                      |
         +------------+------------+
         |            |            |
    MoE Age      MoE Genre    MoE Ethnicite
   (4 experts)  (4 experts)   (4 experts)
         |            |            |
      Linear      Sigmoid      Softmax
```

Chaque tete MoE contient :
- 4 experts (MLP independants)
- 1 gating network (softmax) qui pondere les experts dynamiquement

Entrainement en 3 phases : warmup (25 ep.) + fine-tune partiel (20 ep.) + fine-tune complet (15 ep.)

Loss weights adaptes : age=0.3, genre=1.5, ethnicite=2.0 (priorite genre/ethnicite)

Notebooks : `training/2_modele_multitache.ipynb`

### Strategie 3 — Transfert de connaissances (comparaison de backbones)

Comparaison de 4 architectures pre-entrainees sur ImageNet :

| Backbone | Params | Pre-entraine sur |
|---|---|---|
| MobileNetV2 | ~3.4M | ImageNet |
| EfficientNetB0 | ~5.3M | ImageNet |
| ResNet50 | ~25.6M | ImageNet |
| MobileNetV3Large | ~4.2M | ImageNet |

Notebooks : `training/1_comparaison_backbones.ipynb`

---

## Comparaison des backbones

Resultats de la comparaison rapide (8 warmup + 7 fine-tune epochs par backbone) :

| Backbone | Genre Acc | Genre AUC | Eth Acc | Eth AUC | Age MAE | Temps |
|---|---|---|---|---|---|---|
| **EfficientNetB0** | **90.1%** | **96.4%** | **68.3%** | **90.0%** | 6.19 | 15 min |
| ResNet50 | 89.9% | 96.5% | 66.9% | 90.2% | 5.87 | 33 min |
| MobileNetV3Large | 88.5% | 95.3% | 67.8% | 89.2% | 7.11 | 11 min |
| MobileNetV2 | 85.0% | 93.2% | 55.7% | 83.1% | 7.59 | 13 min |

**EfficientNetB0** est selectionne comme meilleur backbone : meilleur compromis genre/ethnicite avec un temps d'entrainement raisonnable et une taille de modele contenue (~5.3M params).

Le score composite utilise est pondere pour privilegier le genre et l'ethnicite (x3) par rapport a l'age (x1).

### Resultats finaux — Modele Multitache MoE (EfficientNetB0)

Entrainement complet en 3 phases (warmup + fine-tune partiel + fine-tune etendu) :

| Tache | Metrique | Resultat |
|---|---|---|
| **Genre** | Accuracy | **90.7%** |
| | AUC | **96.4%** |
| | AP | 95.3% |
| | ARI | 0.663 |
| | NMI | 0.554 |
| **Ethnicite** | Accuracy | **70.8%** |
| | AUC | **90.6%** |
| | AP | 80.8% |
| | ARI | 0.452 |
| | NMI | 0.408 |
| **Age** | MAE | **6.36 ans** |
| | MSE | 86.3 |
| | R² | **0.778** |

### Resultats — 3 Modeles Individuels (EfficientNetB0)

Chaque modele specialise est entraine independamment avec le meme backbone :

| Modele | Metrique principale | Resultat |
|---|---|---|
| Age | MAE | ~6.2 ans |
| Genre | Accuracy | ~90% |
| Ethnicite | Accuracy | ~70% |

---

## Interpretabilite du modele (Grad-CAM)

Un notebook dedie (`training/4_interpretabilite_gradcam.ipynb`) permet de visualiser **ce que le modele voit** pour chaque prediction grace a Grad-CAM (Gradient-weighted Class Activation Mapping).

Le modele n'utilise ni infrarouge, ni capteur special : il analyse uniquement les **pixels de l'image**. Grad-CAM revele les zones de l'image que le modele considere comme importantes :

- **Genre** : machoire, pilosite faciale, structure osseuse, cheveux
- **Age** : rides, texture de peau, cheveux gris, contour du visage
- **Ethnicite** : teint, forme du visage, traits caracteristiques

Ce notebook ne necessite **aucun entrainement** — il charge les modeles deja entraines et genere les cartes de chaleur.

---

## Donnees et preprocessing

### Dataset : UTKFace

- **23 708 images** de visages recadres (200x200 pixels)
- **Labels** : age (0-116), genre (0=Homme, 1=Femme), ethnicite (0=White, 1=Black, 2=Asian, 3=Indian, 4=Other)
- **Split** : 70% train / 15% validation / 15% test (stratifie sur l'ethnicite)

### Preprocessing

- Redimensionnement a 224x224 pixels
- **Data augmentation** : flip horizontal, crop aleatoire, variations de luminosite/contraste/saturation
- **Focal Loss** pour l'ethnicite : compense le desequilibre des classes (poids inversement proportionnels a la frequence)
- **Mixed precision float16** : accelere l'entrainement sur GPU

### Detection de visage (inference)

L'application utilise **MediaPipe Face Detection** pour isoler le visage avant prediction.
Les modeles ayant ete entraines sur UTKFace (visages deja recadres), cette etape est obligatoire.

En mode temps reel, un guide ovale aide l'utilisateur a positionner son visage. Un rectangle vert apparait quand un visage est detecte.

---

## Application Android

### Fonctionnalites implementees

| Fonctionnalite | Description | Statut |
|---|---|---|
| Capture camera | Prise de photo via CameraX (front/back) | Fait |
| Galerie | Chargement d'image depuis la galerie | Fait |
| Temps reel (bonus) | Prediction en continu via la camera | Fait |
| Authentification | Email/mot de passe + Google Sign-In (Firebase) | Fait |
| Gestion de compte | Modification, suppression, deconnexion | Fait |
| Historique | Liste des predictions passees (Firestore) | Fait |
| Infos modele | Type, precision, metriques du modele utilise | Fait |
| Tutorial | Ecran d'accueil pour les nouveaux utilisateurs | Fait |
| Mode sombre | Theme sombre complet avec toggle dans les parametres | Fait |
| Profil avance | Statistiques, infos securite, modifier le nom, export JSON | Fait |

### Modes de prediction

L'application propose 3 modes de prediction selectionnables dans les parametres :

| Mode | Architecture | Description |
|---|---|---|
| **Hybride** (recommande) | 3x EfficientNetB0 + TTA | Combine les 3 modeles specialises avec flip horizontal pour des resultats plus stables |
| 3 Modeles specialises | 3x EfficientNetB0 | Un modele dedie par tache (age, genre, ethnicite) |
| Multitache MoE | EfficientNetB0 + Mixture of Experts | 1 modele, 4 experts par tache, gating network |

### Modeles TFLite embarques

Tous les modeles utilisent le backbone EfficientNetB0 (224x224) re-entraine sur UTKFace.

| Fichier | Taille | Description |
|---|---|---|
| `age_v2_model.tflite` | 8.4 MB | Age specialise (EfficientNetB0) |
| `gender_v2_model.tflite` | 8.4 MB | Genre specialise (EfficientNetB0) |
| `ethnicity_v2_model.tflite` | 8.4 MB | Ethnicite specialise (EfficientNetB0) |
| `moe_mobilenetv3.tflite` | 17 MB | Multitache MoE (EfficientNetB0 + 4 experts) |
| `face_detection_short_range.tflite` | 0.2 MB | Detection de visage MediaPipe |

---

## Installation et utilisation

### Pre-requis

- Python 3.10+ avec pip
- GPU NVIDIA recommande (CUDA) pour l'entrainement
- Android Studio (pour l'application mobile)
- Dataset UTKFace (telechargeable via Kaggle)

### 1. Cloner le depot

```bash
git clone https://github.com/optmlako2004/face-analysis-deep-learning.git
cd face-analysis-deep-learning
```

### 2. Environnement Python (entrainement)

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Telecharger le dataset UTKFace

Placer les images dans `data/UTKFace/`. Le dataset est disponible sur Kaggle (`jangedoo/utkface-new`).

### 4. Lancer les notebooks d'entrainement

Les notebooks sont dans `training/` et doivent etre executes dans l'ordre :

```
1_comparaison_backbones.ipynb      -> Compare 4 backbones, determine le meilleur
2_modele_multitache.ipynb          -> Entraine le modele multi-tache MoE complet
3_modeles_individuels.ipynb        -> Entraine les 3 modeles specialises
4_interpretabilite_gradcam.ipynb   -> Visualise ce que le modele "voit" (Grad-CAM)
```

Chaque notebook est **independant** et peut etre execute separement.
Les resultats du notebook 1 (meilleur backbone) sont automatiquement utilises par les notebooks 2 et 3.

### 5. Telecharger l'APK

L'APK pre-construite est disponible dans la section [Releases](https://github.com/optmlako2004/face-analysis-deep-learning/releases) du depot. Cliquer sur `FacePredictor.apk` pour la telecharger et l'installer sur un appareil Android.

### 6. Builder l'APK soi-meme (optionnel)

```bash
cd FacePredictor
./gradlew assembleDebug         # Genere l'APK debug
```

L'APK generee se trouve dans `FacePredictor/app/build/outputs/apk/debug/app-debug.apk`.

Ou via Android Studio :
1. Ouvrir le dossier `FacePredictor/` dans Android Studio
2. Synchroniser Gradle
3. Configurer Firebase (voir ci-dessous)
4. Build > Run sur un appareil ou emulateur

### 6. Configuration Firebase

1. Creer un projet sur [Firebase Console](https://console.firebase.google.com)
2. Activer Authentication (Email/mot de passe + Google)
3. Activer Firestore Database
4. Creer l'index composite : collection `predictions`, champs `userId` (ASC) + `createdAt` (DESC)
5. Telecharger `google-services.json` dans `FacePredictor/app/`

---

## Organisation du depot

```
face-analysis-deep-learning/
├── training/                           # Notebooks d'entrainement principaux
│   ├── 1_comparaison_backbones.ipynb   #   Strategie 3 : comparaison des backbones
│   ├── 2_modele_multitache.ipynb       #   Strategie 2 : modele multi-tache MoE
│   ├── 3_modeles_individuels.ipynb     #   Strategie 1 : 3 modeles specialises
│   ├── 4_interpretabilite_gradcam.ipynb #   Visualisation Grad-CAM (ce que voit le modele)
│   └── output_v2/                      #   Modeles et resultats generes
├── notebooks/                          # Notebooks d'exploration et anciennes versions
│   ├── age/                            #   Modeles de regression d'age (V1, V2, V3)
│   ├── gender/                         #   Modeles de classification de genre
│   ├── ethnicity/                      #   Modeles de classification d'ethnicite
│   ├── multitask/                      #   Modeles multi-taches (V1 a V5)
│   ├── moe/                            #   Mixture of Experts MobileNetV3
│   └── experiments/                    #   Experiences (RGB vs grayscale, Grad-CAM, etc.)
├── artifacts/                          # Artefacts d'entrainement (metriques, courbes, modeles)
│   ├── age/ gender/ ethnicity/         #   Par tache
│   ├── multitask/ moe/                 #   Multi-taches et MoE
│   └── experiments/                    #   Visuels et resultats des experiences
├── FacePredictor/                      # Application Android (Kotlin)
│   └── app/src/main/
│       ├── java/.../facepredictor/     #   Code source (ui/, ml/, data/, utils/)
│       ├── assets/                     #   Modeles TFLite embarques
│       └── res/                        #   Layouts, drawables, strings
├── tests/                              # Scripts de test de prediction
├── diagnostics/                        # Scripts d'analyse et profiling
├── data/                               # Dataset UTKFace (non versionne)
├── config.py                           # Chemins et constantes partagees
├── requirements.txt                    # Dependances Python
└── README.md
```

---

## Metriques et evaluation

Toutes les metriques demandees par le sujet sont calculees :

### Classification (genre, ethnicite)

| Metrique | Description |
|---|---|
| **Accuracy** | Pourcentage de predictions correctes |
| **AUC** (Area Under ROC Curve) | Capacite a distinguer les classes |
| **AP** (Average Precision) | Precision moyenne ponderee |
| **ARI** (Adjusted Rand Index) | Concordance ajustee par le hasard |
| **NMI** (Normalized Mutual Information) | Information mutuelle normalisee |

### Regression (age)

| Metrique | Description |
|---|---|
| **MAE** (Mean Absolute Error) | Erreur moyenne en annees |
| **MSE** (Mean Squared Error) | Erreur quadratique moyenne |
| **R²** (Coefficient de determination) | Proportion de variance expliquee |

---

## Livrables

| Livrable | Statut | Emplacement |
|---|---|---|
| Code source complet | Fait | Ce depot GitHub |
| Application Android (.apk) | Fait | [Telecharger dans Releases](https://github.com/optmlako2004/face-analysis-deep-learning/releases) |
| 3 modeles specialises (Strategie 1) | Fait | `training/3_modeles_individuels.ipynb` |
| Modele multi-tache MoE (Strategie 2) | Fait | `training/2_modele_multitache.ipynb` |
| Comparaison des backbones (Strategie 3) | Fait | `training/1_comparaison_backbones.ipynb` |
| Detection de visage (MediaPipe) | Fait | Integre dans l'app |
| Authentification (Firebase) | Fait | Email + Google Sign-In |
| Historique des predictions | Fait | Firestore |
| Mode temps reel (bonus) | Fait | Camera en continu |
| README | Fait | Ce fichier |
| Rapport technique | Fait | `rapport_SAE.pdf` |
| Slides de presentation | Fait | `presentation_SAE.pdf` |
| Video de demonstration | Fait | Demo live a la soutenance |

---

## Soutenance

- **Date** : Vendredi 3 avril 2026, entre 9h et 13h
- **Lieu** : Salle S108
- **Format** : 15 minutes de presentation + 5 minutes de questions

---

_Projet SAE BUT3 Informatique — 2025-2026_
_Encadrement : Bilal Faye (LIPN, CNRS UMR 7030, Universite Sorbonne Paris Nord)_
