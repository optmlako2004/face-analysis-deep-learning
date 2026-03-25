git clone https://github.com/VOTRE_USERNAME/SAE.git
# FacePredictor — Deep Learning + Android

Projet SAE (BUT 3 Informatique) : application Android de prediction d'age, de genre et d'ethnicite a partir d'une photo de visage, appuyee par des modeles TensorFlow / TensorFlow Lite (MobileNetV3, EfficientNet, Mixture of Experts).

## Organisation du depot

```
face-analysis-deep-learning/
├── notebooks/
│   ├── age/                          # Modeles et regression d'age
│   ├── gender/                       # Classification de genre
│   ├── ethnicity/                    # Classification d'ethnicite
│   ├── multitask/                    # Modeles unifies multi-taches
│   ├── experiments/                  # Comparaisons backbones, viz
│   └── moe/train_moe.ipynb           # Entrainement MoE MobileNetV3
├── artifacts/                        # Modeles, metriques, visuels tries par tache/notebook
│   ├── age/ gender/ ethnicity/ ...   # Artefacts d'entrainement par tache
│   ├── multitask/ moe/               # Artefacts multi-taches et MoE
│   └── experiments/<notebook>/{img,results}/
│                                     # Visuels/JSON generes par chaque notebook d'experiences
├── artifacts/experiments/tests/img/  # Images de test manuelles
├── tests/                            # Scripts de tests de prediction
├── diagnostics/                      # Scripts d'analyse/profiling MoE
├── FacePredictor/                    # Application Android (Kotlin)
├── data/                             # Jeux de donnees locaux (non versionnes)
├── config.py                         # Constantes/chemins communs
├── requirements.txt
└── README.md
```

### Notebooks (reorganises)
- `notebooks/age/` : construction et regression d'age (V2 EfficientNet, MobileNet V3, baseline).
- `notebooks/gender/` : classification du genre (V2 EfficientNet, MobileNet V3).
- `notebooks/ethnicity/` : classification d'ethnicite (V2 EfficientNet, MobileNet V3).
- `notebooks/multitask/` : modeles multi-taches (versions V2, V3, V4 equilibree, V5 MobileNet, MobileNet multi-tache).
- `notebooks/experiments/` : comparaison MobileNet vs EfficientNet, RGB vs grayscale, visualisation des intervalles d'age et filtres.
- `notebooks/moe/train_moe.ipynb` : entrainement complet du modele Mixture of Experts (MobileNetV3 backbone + gating networks).

### Artefacts ML
- `artifacts/<tache>/` : artefacts d'entrainement par tache (JSON de metriques, PNG de courbes, TFLite/Keras exportes).
- `artifacts/experiments/<notebook>/img` : visuels produits par chaque notebook d'experiences.
- `artifacts/experiments/<notebook>/results` : resultats/rapports JSON par notebook d'experiences.
- `artifacts/experiments/tests/img` : images de test manuelles.
- `data/` : dossier ignore (ex. `data/UTKFace/`). Deposez ici vos donnees et splits.

## Prise en main Python (entrainement/benchmark)
1) Creer l'environnement :
```bash
cd face-analysis-deep-learning
python -m venv .venv && source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```
2) Placer le dataset UTKFace (ou votre dataset equivalent) dans `data/UTKFace/`.
3) Lancer un notebook d'entrainement/etude depuis `notebooks/` (Jupyter ou VS Code) ou executer les scripts de test rapides :
- `python tests/test_all_modes.py` : compare Oriente V2, Multitache V4 et Hybride sur un echantillon equilibre.
- `python tests/test_hybrid_mode.py` : validation du mode hybride (ensemblage + TTA).
- `python tests/test_efficientnet_models.py` : verification des modeles EfficientNet.
- `python tests/test_moe_comparison.py` : comparaison MoE vs autres modes.

Tous les notebooks commencent par une cellule *Bootstrap* qui remet le `cwd` a la racine du projet et importe `config.py` (chemins, assets Android, helper Kaggle). Executer cette premiere cellule avant les suivantes pour eviter les problemes de chemins apres la reorganisation.

## Application Android (Kotlin)
- Code : `FacePredictor/app/src/main/java/com/sae/facepredictor/` (modules `ui/`, `data/`, `ml/`).
- Modeles embarques : `FacePredictor/app/src/main/assets/` (MoE + EfficientNet/MobileNet + Face Detection MediaPipe).
- Firebase : `google-services.json` deja present; verifier vos credentiels et SHA si vous changez d'environnement.
- Build rapide en ligne de commande :
```bash
cd FacePredictor
./gradlew assembleDebug    # APK debug
./gradlew assembleRelease  # APK release (requiert keystore)
```
- Build/Run via Android Studio : ouvrir `FacePredictor/`, laisser Gradle synchroniser, choisir un appareil/emulateur, puis `Run`.

## Modes de prediction exposes dans l'app
- **MoE Expert (recommande)** : MobileNetV3 + 3 experts par tache; meilleur compromis vitesse/precision.
- **Hybride** : ensemble genre V2 + age V2 + ethnicite depuis multitache V4 avec TTA.
- **Oriente V2** : trois modeles specialises separes (genre/age/ethnicite V2).
- **Multitache V4/V5** : un seul modele unifie pour les trois taches.

## Ressources complementaires
- Comparaisons backbones (MobileNet vs EfficientNet), impact RGB/Grayscale, et visualisation des filtres : voir `notebooks/experiments/` + images dans `artifacts/`.
- Analyse MoE (courbes, matrices, distribution dataset) : `artifacts/moe_mobilenetv3/`.

## Bonnes pratiques
- Garder `data/` hors du versionning (datasets volumineux/sensibles).
- Regenerer les assets TFLite apres tout retrain (copier le .tflite dans `FacePredictor/app/src/main/assets/`).
- Verifier la coherence des dependances : `requirements.txt` pour Python, `build.gradle.kts` pour Android.

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
