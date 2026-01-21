# Application Mobile Intelligente - Prediction d'Age, Genre et Ethnicite

## Vue d'Ensemble du Projet

**SAE BUT 3 Informatique - Annee 2025-2026**

Ce projet developpe une **application mobile Android intelligente** capable de predire l'age, le genre et l'ethnicite d'une personne a partir de son visage, en utilisant des modeles de Deep Learning optimises et deployes avec TensorFlow Lite.

### Fonctionnalites de l'Application

| Fonctionnalite       | Description                                    | Statut |
| -------------------- | ---------------------------------------------- | ------ |
| Capture camera       | Prendre une photo via la camera                | Fait   |
| Import galerie       | Charger une photo existante                    | Fait   |
| Detection de visage  | MediaPipe Face Detection                       | Fait   |
| Prediction IA        | Age, Genre, Ethnicite                          | Fait   |
| Switch de modele     | 3 modes: Hybride, Oriente V2, Multitache V4    | Fait   |
| Authentification     | Firebase Auth (Email + Google Sign-In)         | Fait   |
| Historique           | Sauvegarder les predictions (Firestore)        | Fait   |
| Page infos modele    | Affichage dynamique des metriques              | Fait   |
| Temps reel           | Prediction en streaming camera                 | Bonus  |

---

## Demarrage Rapide (Installation)

### Prerequis

- Python 3.10 ou superieur (recommande: 3.11 ou 3.12)
- pip (gestionnaire de paquets Python)
- Git
- 8 Go de RAM minimum (16 Go recommande)
- GPU NVIDIA avec CUDA 12.x (optionnel, accelere l'entrainement)

### Etape 1 : Cloner le projet

```bash
git clone https://github.com/VOTRE_USERNAME/SAE.git
cd SAE
```

### Etape 2 : Creer un environnement virtuel (OBLIGATOIRE)

**IMPORTANT** : Toujours utiliser un environnement virtuel pour eviter les conflits de dependances !

```bash
# Creer l'environnement virtuel
python -m venv .venv

# Activer l'environnement
# Sur Linux/macOS :
source .venv/bin/activate

# Sur Windows (PowerShell) :
.venv\Scripts\Activate.ps1

# Sur Windows (CMD) :
.venv\Scripts\activate.bat
```

### Etape 3 : Installer les dependances

```bash
# Mettre a jour pip d'abord (evite des erreurs)
pip install --upgrade pip

# Installer toutes les dependances
pip install -r requirements.txt
```

### Etape 4 : Verifier l'installation

```bash
# Verifier que TensorFlow est installe
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__}')"

# Verifier si un GPU est disponible (optionnel)
python -c "import tensorflow as tf; print('GPU:', tf.config.list_physical_devices('GPU'))"
```

### Etape 5 : Telecharger le dataset UTKFace

```bash
# Creer le dossier data
mkdir -p data

# Option 1: Depuis Kaggle (necessite un compte)
pip install kaggle
kaggle datasets download -d jangedoo/utkface-new
unzip utkface-new.zip -d data/

# Option 2: Telechargement manuel
# Telecharger depuis https://www.kaggle.com/datasets/jangedoo/utkface-new
# Extraire dans data/UTKFace/
```

### Etape 6 : Lancer Jupyter

```bash
jupyter notebook
```

### Ordre d'Execution des Cellules (IMPORTANT)

**Les notebooks doivent etre executes cellule par cellule, dans l'ordre !**

1. Executer d'abord les cellules d'imports et configuration
2. Ne pas sauter de cellules (chaque cellule depend des precedentes)
3. Si une erreur survient, redemarrer le kernel et recommencer

**Note technique** : Les notebooks utilisent `CosineDecay` pour le learning rate.
Ne pas ajouter `ReduceLROnPlateau` dans les callbacks car ces deux mecanismes sont incompatibles.

### Problemes Courants et Solutions

| Probleme                          | Cause                            | Solution                                           |
| --------------------------------- | -------------------------------- | -------------------------------------------------- |
| `ModuleNotFoundError`             | Environnement virtuel non active | Executer `source .venv/bin/activate`               |
| `ModuleNotFoundError: cv2`        | OpenCV manquant                  | `pip install opencv-python`                        |
| Conflits de versions              | Installation globale existante   | Supprimer `.venv/` et recreer                      |
| `CUDA error`                      | Drivers GPU incompatibles        | Utiliser CPU ou mettre a jour CUDA                 |
| `Memory error`                    | RAM insuffisante                 | Reduire `BATCH_SIZE` a 16                          |
| `TypeError: LearningRateSchedule` | Conflit CosineDecay/ReduceLR     | Ne pas utiliser ReduceLROnPlateau avec CosineDecay |

---

## Historique de l'Evolution des Modeles

Cette section documente l'evolution de nos modeles multi-taches, les problemes rencontres et les solutions apportees.

### Tableau Comparatif des Performances

| Version | Age MAE  | Genre Acc | Ethnicite Acc | Eth. F1 macro | Statut   |
| ------- | -------- | --------- | ------------- | ------------- | -------- |
| **V1**  | 6.66 ans | 82.00%    | 62.43%        | 0.4614        | Baseline |
| **V2**  | 5.78 ans | 81.95%    | 59.80%        | 0.5109        | Ameliore |
| **V3**  | 5.88 ans | 49.54%    | 51.43%        | 0.4720        | Echec    |
| **V4**  | En cours | En cours  | En cours      | En cours      | Test     |

---

### V1 : Modele de Base (`multitask_model.ipynb`)

**Architecture initiale** avec EfficientNetB0 et trois tetes specialisees.

#### Configuration V1

| Parametre         | Valeur                        |
| ----------------- | ----------------------------- |
| Backbone          | EfficientNetB0 (ImageNet)     |
| Couches partagees | Dense(512)                    |
| Heads             | Dense(128) simple             |
| Loss Age          | Huber (delta=5.0)             |
| Loss Genre        | BinaryCrossentropy            |
| Loss Ethnicite    | SparseCategoricalCrossentropy |
| Optimizer         | Adam                          |
| Warmup epochs     | 15                            |
| Finetune epochs   | 40                            |
| Couches degelees  | 50                            |

#### Resultats V1

```
Age:
   MAE: 6.66 annees
   RMSE: 9.87 annees

Genre:
   Accuracy: 82.00%
   F1-Score: 0.7962

Ethnicite:
   Accuracy: 62.43%
   F1-Score (macro): 0.4614
   F1-Score (weighted): 0.5794
```

#### Problemes identifies dans V1

1. **Classe "Others" completement ignoree** : Precision et Recall = 0%
2. **Fort desequilibre des classes** : White (~10k) vs Others (~1.7k)
3. **F1-Score ethnicite tres bas** : Le modele favorise les classes majoritaires
4. **Heads trop simples** : Une seule couche Dense(128) par tache

---

### V2 : Ameliorations Architecturales (`multitask_model_v2.ipynb`)

**Objectif** : Ameliorer la gestion du desequilibre des classes et la capacite du modele.

#### Changements V1 -> V2

| Aspect                 | V1                            | V2                                         | Raison                         |
| ---------------------- | ----------------------------- | ------------------------------------------ | ------------------------------ |
| **Architecture heads** | Dense(128)                    | Dense(256)->Dense(128)                     | Plus de capacite               |
| **Loss ethnicite**     | SparseCategoricalCrossentropy | **Focal Loss** (gamma=2.0) + Class Weights | Gestion du desequilibre        |
| **Loss genre**         | BinaryCrossentropy            | + **Label Smoothing 0.1**                  | Moins d'overfitting            |
| **Loss age**           | Huber(delta=5.0)              | Huber(delta=**3.0**)                       | Plus sensible aux erreurs      |
| **Loss weights**       | Tous a 1.0                    | Age=**2.0**, Genre=1.0, Eth=**2.0**        | Equilibre les taches           |
| **Attention**          | Aucune                        | **Squeeze-Excitation**                     | Focus sur features importantes |
| **Optimizer**          | Adam                          | **AdamW** + weight decay                   | Meilleure regularisation       |
| **Couches degelees**   | 50                            | **70**                                     | Plus de fine-tuning            |
| **Epochs**             | 15+40                         | **20+60**                                  | Convergence complete           |
| **Couches partagees**  | Dense(512)                    | **Dense(1024)->Dense(512)**                | Plus de representation         |

#### Qu'est-ce que la Focal Loss ?

La **Focal Loss** reduit l'importance des exemples "faciles" pour forcer le modele a apprendre les exemples "difficiles" (classes minoritaires).

```
Focal Loss = -(1 - p)^gamma * log(p)

Avec gamma=2.0:
- Si p=0.9 (facile): poids = 0.01 (ignore)
- Si p=0.1 (difficile): poids = 0.81 (focus)
```

#### Qu'est-ce que Squeeze-Excitation ?

C'est un mecanisme d'**attention par canal** qui apprend quels canaux de features sont importants pour la tache.

```
Input (H, W, C)
    |
Global Average Pooling -> (1, 1, C)
    |
Dense(C/16) -> ReLU -> Dense(C) -> Sigmoid
    |
Scale: Input * Attention Weights
```

#### Resultats V2

```
Age:
   MAE: 5.78 annees (amelioration de 13%)
   RMSE: 8.41 annees

Genre:
   Accuracy: 81.95%
   F1-Score: 0.8009

Ethnicite:
   Accuracy: 59.80%
   F1-Score (macro): 0.5109 (amelioration de 10%)
   F1-Score (weighted): 0.6008
```

#### Analyse V2

**Ameliorations** :

- Age MAE : 6.66 -> 5.78 (-13%)
- F1-Score ethnicite macro : 0.46 -> 0.51 (+10%)
- La classe "Others" a maintenant un recall de 28% (vs 0%)

**Problemes restants** :

- Genre accuracy legerement reduite (82% -> 81.95%)
- Classe "Indian" sous-performante (F1=0.34)
- "Others" toujours difficile (F1=0.21)

---

### V3 : Tentative d'Equilibrage Agressif (`multitask_model_v3_balanced.ipynb`)

**Objectif** : Forcer l'equilibre avec oversampling et Focal Loss sur toutes les taches.

#### Changements V2 -> V3

| Aspect                | V2                       | V3                                       |
| --------------------- | ------------------------ | ---------------------------------------- |
| **Dataset**           | Original (desequilibre)  | **Oversampling** (70% de la classe max)  |
| **Loss genre**        | BCE + Label Smoothing    | **Binary Focal Loss** (gamma=2.0)        |
| **Loss ethnicite**    | Focal Loss               | Focal Loss + **Label Smoothing 0.1**     |
| **Class weights**     | Calcules automatiquement | **Renforces manuellement** (Others x2.0) |
| **Data augmentation** | Standard                 | **Renforcee** (rotation 0.15, zoom 0.2)  |
| **Loss weights**      | Age=2, Genre=1, Eth=2    | Age=0.8, Genre=1.2, Eth=**1.5**          |

#### Resultats V3 - ECHEC

```
Age:
   MAE: 5.88 annees

Genre:
   Accuracy: 49.54% <-- PROBLEME MAJEUR
   F1-Score: 0.6625

   Precision Male: 0.00
   Recall Male: 0.00
   Precision Female: 0.50
   Recall Female: 1.00

Ethnicite:
   Accuracy: 51.43%
   F1-Score (macro): 0.4720
```

#### Analyse de l'Echec V3

**Probleme** : Le modele predit **TOUJOURS "Female"** pour le genre !

**Causes identifiees** :

1. **Binary Focal Loss mal parametree** : Le alpha=0.6 (favorisant Female) combine avec l'oversampling a cree un biais extreme

2. **Oversampling excessif** : Dupliquer les images a reduit la diversite et cree du surapprentissage

3. **Conflit entre regularisations** : Label smoothing + Focal Loss + Class weights = trop de regularisation

4. **Dataset desequilibre artificiellement** : L'oversampling a 70% a perturbe les proportions naturelles

**Lecons apprises** :

- Ne pas combiner trop de techniques de regularisation
- L'oversampling peut etre contre-productif sur les images
- Tester incrementalement chaque changement

---

### V4 : Equilibrage par Sous-Echantillonnage (`multitask_model_v4_balanced.ipynb`)

**Objectif** : Dataset parfaitement equilibre par sous-echantillonnage + architecture simplifiee.

#### Changements V3 -> V4

| Aspect                | V3                                      | V4                                         |
| --------------------- | --------------------------------------- | ------------------------------------------ |
| **Nombre de classes** | 5 (White, Black, Asian, Indian, Others) | **4** (sans Others)                        |
| **Equilibrage**       | Oversampling (duplication)              | **Sous-echantillonnage**                   |
| **Dataset**           | ~38k images                             | ~12.8k images (1.6k/groupe)                |
| **Loss genre**        | Binary Focal Loss                       | **BCE + Label Smoothing** (retour)         |
| **Loss ethnicite**    | Focal Loss                              | **SparseCategoricalCrossentropy** (retour) |
| **Loss weights**      | Age=0.8, Genre=1.2, Eth=1.5             | Age=0.5, Genre=1.5, Eth=1.5                |

#### Justification des Choix V4

1. **Suppression de "Others"** : Cette classe est trop heterogene (melange de plusieurs ethnicites) et perturbe l'apprentissage

2. **Sous-echantillonnage** : Au lieu de dupliquer les minorites (risque de surapprentissage), on reduit les majoritaires (conserve la diversite)

3. **Retour aux losses standard** : Moins de regularisation = moins de risque de comportements inattendus

4. **Focus sur genre et ethnicite** : Loss weight age=0.5 pour prioriser les classifications

#### Resultats V4 (En cours d'evaluation)

Le notebook V4 est pret mais n'a pas encore ete execute completement. Les resultats seront ajoutes apres entrainement.

---

## Modeles Specialises V2

En plus des modeles multi-taches, des modeles specialises V2 ont ete developpes avec des ameliorations specifiques a chaque tache.

### Gender Model V2 (`gender_model_v2.ipynb`)

| Caracteristique   | Description                                               |
| ----------------- | --------------------------------------------------------- |
| **Preprocessing** | CLAHE (Contrast Limited Adaptive Histogram Equalization)  |
| **Loss**          | Binary Focal Loss avec label smoothing                    |
| **Augmentation**  | Rotation moderee (±27°), luminosite (±15%)                |
| **Attention**     | Squeeze-Excitation blocks                                 |
| **Seuil**         | Optimise via courbe ROC (Youden's J)                      |
| **TTA**           | Test Time Augmentation (flip, luminosite, rotation, zoom) |

### Age Model V2 (`age_model_v2.ipynb`)

| Caracteristique  | Description                             |
| ---------------- | --------------------------------------- |
| **Loss**         | Huber Loss (delta=3.0)                  |
| **Augmentation** | Mixup (alpha=0.2), rotation (±45°)      |
| **Attention**    | CBAM (Channel + Spatial Attention)      |
| **Oversampling** | Focus sur les tranches jeunes (0-20) x2 |
| **Epochs**       | 30 warmup + 80 fine-tuning              |

### Ethnicity Model V2 (`ethnicity_model_v2.ipynb`)

| Caracteristique          | Description                               |
| ------------------------ | ----------------------------------------- |
| **Classes**              | 5 classes avec "Others"                   |
| **Systeme de confiance** | Si max_proba < 0.35 → predit "Others"     |
| **Loss**                 | Focal Loss (gamma=3.0) avec class weights |
| **Augmentation**         | Mixup (alpha=0.3), rotation (±45°)        |
| **Label Smoothing**      | 0.15                                      |
| **Epochs**               | 30 warmup + 70 fine-tuning                |

---

## Architecture Technique Detaillee

### Architecture du Modele Multi-Tache

```
                    +-------------------------------------+
                    |         Input (128x128x3)           |
                    +-----------------+-------------------+
                                      |
                    +-----------------v-------------------+
                    |     Data Augmentation (train)       |
                    +-----------------+-------------------+
                                      |
                    +-----------------v-------------------+
                    |   EfficientNetB0 (Backbone partage) |
                    |        Pre-entraine ImageNet        |
                    +-----------------+-------------------+
                                      |
                    +-----------------v-------------------+
                    |   Squeeze-Excitation (V2+)          |
                    +-----------------+-------------------+
                                      |
                    +-----------------v-------------------+
                    |   GlobalAveragePooling2D + GMP      |
                    |        (Features partagees)         |
                    +-----------------+-------------------+
                                      |
                    +-----------------v-------------------+
                    |  Dense 1024 + BN + Dropout (V2+)    |
                    |  Dense 512 + BN + Dropout           |
                    |        (Representation commune)     |
                    +-----------------+-------------------+
                                      |
          +---------------------------+---------------------------+
          |                           |                           |
          v                           v                           v
+-----------------+         +-----------------+         +-----------------+
|   Age Head      |         |  Gender Head    |         | Ethnicity Head  |
|   Dense 256     |         |   Dense 256     |         |   Dense 256     |
|   Dense 128     |         |   Dense 128     |         |   Dense 128     |
|   Dense 1       |         |   Dense 1       |         |   Dense 4/5     |
|   (lineaire)    |         |   (sigmoid)     |         |   (softmax)     |
+-----------------+         +-----------------+         +-----------------+
```

### Pourquoi EfficientNetB0 ?

| Critere           | EfficientNetB0 | VGG16 | ResNet50 |
| ----------------- | -------------- | ----- | -------- |
| Parametres        | ~4M            | ~138M | ~25M     |
| Accuracy ImageNet | 77.1%          | 71.3% | 76.0%    |
| Taille TFLite     | ~5MB           | ~60MB | ~25MB    |
| Inference mobile  | Rapide         | Lent  | Moyen    |

**EfficientNetB0** offre le meilleur compromis performance/taille pour le deploiement mobile.

### Strategie d'Entrainement en 2 Phases

#### Phase 1 : Warmup (Backbone Gele)

```
Duree: 15-20 epochs
Learning Rate: 1e-3 (eleve)
Backbone: GELE (non-entrainable)
Objectif: Entrainer les heads sans perturber les features ImageNet
```

#### Phase 2 : Fine-tuning (Backbone Partiellement Degele)

```
Duree: 40-60 epochs
Learning Rate: 1e-5 (tres faible)
Backbone: 50-70 dernieres couches degelees
Objectif: Adapter les features aux visages
```

### Data Augmentation

```python
data_augmentation = keras.Sequential([
    RandomFlip("horizontal"),      # Miroir horizontal
    RandomRotation(0.1-0.15),      # Rotation +/-10-15%
    RandomZoom(0.15-0.2),          # Zoom +/-15-20%
    RandomTranslation(0.1, 0.1),   # Decalage +/-10%
    RandomBrightness(0.2),         # Luminosite +/-20%
    RandomContrast(0.2),           # Contraste +/-20%
])
```

---

## Pipeline de Donnees

### Format d'Entree

- Taille : **128x128 pixels**
- Format : **RGB** (3 canaux)
- Valeurs : **[0, 255]** (non normalisees - EfficientNet normalise en interne)

### Dataset UTKFace

| Statistique  | Valeur        |
| ------------ | ------------- |
| Total images | ~23,700       |
| Age min/max  | 1-116 ans     |
| Age moyen    | 33 ans        |
| White        | ~10,000 (42%) |
| Black        | ~4,500 (19%)  |
| Asian        | ~3,400 (14%)  |
| Indian       | ~3,900 (16%)  |
| Others       | ~1,700 (7%)   |

### Gestion du Desequilibre

| Technique     | Description                   | Utilisee dans |
| ------------- | ----------------------------- | ------------- |
| Class Weights | Poids inverses a la frequence | V1, V2, V3    |
| Focal Loss    | Focus sur exemples difficiles | V2, V3        |
| Oversampling  | Duplication des minorites     | V3            |
| Undersampling | Reduction des majoritaires    | V4            |

---

## Detection de Visage (OBLIGATOIRE)

**IMPORTANT** : Les modeles ont ete entraines sur UTKFace (visages recadres). Sans detection de visage prealable, les predictions seront incorrectes !

```
SANS detection :                    AVEC detection :
+----------------+                    +----------------+
|                |                    |    +------+    |
|      Person    |  Visage = 10%      |    | Face |    |  Visage = 100%
|     /|\        |  de l'image        |    +------+    |  de l'entree
|     / \        |  --> Prediction    |                |  --> Prediction
+----------------+    incorrecte      +----------------+    correcte
```

### Options de Detection

| Solution                 | Latence Mobile | Recommandation |
| ------------------------ | -------------- | -------------- |
| MediaPipe Face Detection | ~15-30ms       | Recommande     |
| ML Kit (Firebase)        | ~20-40ms       | Bon choix      |
| MTCNN                    | ~100-200ms     | Plus lent      |
| OpenCV Haar Cascade      | ~30-50ms       | Moins precis   |

---

## Structure du Projet

```
SAE/
|
|-- Notebooks d'Entrainement
|   |-- age_prediction_model.ipynb       # Modele age specialise V1
|   |-- age_model_v2.ipynb               # Modele age specialise V2 (ameliore)
|   |-- gender_prediction_model.ipynb    # Modele genre specialise V1
|   |-- gender_model_v2.ipynb            # Modele genre specialise V2 (ameliore)
|   |-- ethnicity_prediction_model.ipynb # Modele ethnicite specialise V1
|   |-- ethnicity_model_v2.ipynb         # Modele ethnicite specialise V2 (ameliore)
|   |-- multitask_model.ipynb            # Multi-tache V1
|   |-- multitask_model_v2.ipynb         # Multi-tache V2 (ameliore)
|   |-- multitask_model_v3_balanced.ipynb# Multi-tache V3 (echec)
|   |-- multitask_model_v4_balanced.ipynb# Multi-tache V4 (en cours)
|   |-- utkface_analysis.ipynb           # Analyse exploratoire
|
|-- Notebooks de Cours/Reference
|   |-- Cours_3__Peceptron.ipynb
|   |-- Cours 1_ Algebre.ipynb
|   |-- Cours 2_ Optimisation.ipynb
|   |-- Cours 4_ MLP.ipynb
|   |-- Cours_5__CNN.ipynb
|
|-- artifacts/                           # Modeles entraines
|   |-- multitask_model_final.keras      # V1
|   |-- multitask_v2_model_final.keras   # V2
|   |-- multitask_v3_final.keras         # V3
|   |-- multitask_model.tflite           # Pour mobile
|   |-- gender_v2_model_final.keras      # Genre V2
|   |-- age_v2_model_final.keras         # Age V2
|   |-- ethnicity_v2_model_final.keras   # Ethnicite V2
|   |-- *_model_info.json                # Metadonnees
|
|-- data/UTKFace/                        # Dataset (non versionne)
|
|-- FacePredictor/                       # Application Android (voir section dediee)
|
|-- README.md                            # Cette documentation
|-- requirements.txt                     # Dependances Python
|-- Sujet SAE 1.pdf                      # Enonce du projet
```

---

## Fichiers Generes

Apres entrainement, chaque modele genere :

| Fichier                    | Description                      |
| -------------------------- | -------------------------------- |
| `*_model_best.keras`       | Meilleur checkpoint (val_loss)   |
| `*_model_final.keras`      | Modele final                     |
| `*_model.tflite`           | Version mobile optimisee         |
| `*_model_info.json`        | Metadonnees (classes, metriques) |
| `*_training_history.png`   | Courbes d'entrainement           |
| `*_confusion_matrices.png` | Matrices de confusion            |

### Exemple de `model_info.json`

```json
{
  "model_type": "multitask_v2",
  "version": "2.0",
  "img_size": 128,
  "input_range": [0, 255],
  "outputs": {
    "age": { "type": "regression", "output_name": "age_output" },
    "gender": {
      "type": "binary_classification",
      "classes": ["Male", "Female"],
      "threshold": 0.5
    },
    "ethnicity": {
      "type": "multiclass_classification",
      "classes": ["White", "Black", "Asian", "Indian", "Others"]
    }
  },
  "metrics": {
    "age_mae": 5.78,
    "gender_accuracy": 0.8195,
    "ethnicity_accuracy": 0.598
  }
}
```

---

## Application Android - FacePredictor

### Presentation

L'application **FacePredictor** est une application Android native (Kotlin) qui permet de predire l'age, le genre et l'ethnicite d'une personne a partir de son visage.

### Technologies Utilisees

| Technologie       | Version | Usage                           |
| ----------------- | ------- | ------------------------------- |
| Kotlin            | 1.9.20  | Langage principal               |
| Android SDK       | 34      | Target SDK                      |
| TensorFlow Lite   | 2.17.0  | Inference des modeles           |
| MediaPipe         | 0.10.9  | Detection de visage             |
| Firebase Auth     | BoM 33  | Authentification                |
| Firestore         | BoM 33  | Stockage des predictions        |
| CameraX           | 1.3.1   | Capture photo                   |
| Coil              | 2.5.0   | Chargement d'images             |

### Modes de Prediction

L'application offre **trois modes de prediction** selectionnables via un RadioGroup :

| Mode              | Description                                | Modeles utilises                                 |
| ----------------- | ------------------------------------------ | ------------------------------------------------ |
| **Hybride**       | Combine le meilleur des 2 approches        | Gender V2 + Ethnicity V4 + Age V2                |
| **Oriente V2**    | 3 modeles specialises independants         | age_v2, gender_v2, ethnicity_v2                  |
| **Multitache V4** | 1 modele unifie pour les 3 taches          | multitask_model (EfficientNetB0 multi-output)    |

#### Comparaison des Performances (sur 40 images test)

| Mode              | Genre   | Ethnicite | Age MAE  |
| ----------------- | ------- | --------- | -------- |
| **Hybride**       | 97.5%   | 72.5%     | 5.62 ans |
| **Oriente V2**    | 97.5%   | 30.0%     | 5.62 ans |
| **Multitache V4** | 77.5%   | 72.5%     | 5.22 ans |

**Recommandation** : Le mode **Hybride** est recommande car il combine les forces des deux approches :
- Genre depuis V2 Oriente (97.5% de precision)
- Ethnicite depuis V4 Multitache (72.5% de precision)
- Age depuis V2 Oriente

#### Analyse Comparative - Pourquoi le Mode Hybride ?

Lors des tests, nous avons observe que chaque approche avait des forces et faiblesses complementaires :

**Test du Mode Oriente V2 (3 modeles separes)** :
```
Image 1: Femme Blanche 25 ans → Genre: Femme ✓ | Ethnicite: Noir ✗
Image 2: Homme Noir 35 ans    → Genre: Homme ✓ | Ethnicite: Blanc ✗
Image 3: Homme Asiatique 45   → Genre: Homme ✓ | Ethnicite: Indien ✗
Image 4: Femme Indienne 28    → Genre: Femme ✓ | Ethnicite: Asiatique ✗
Image 5: Homme Blanc 55 ans   → Genre: Homme ✓ | Ethnicite: Blanc ✓

Resultat: Genre 5/5 (100%) | Ethnicite 1/5 (20%)
Constat: Excellent pour le genre, mauvais pour l'ethnicite
```

**Test du Mode Multitache V4 (1 modele unifie)** :
```
Image 1: Femme Blanche 25 ans → Genre: Femme ✓ | Ethnicite: Blanc ✓
Image 2: Homme Noir 35 ans    → Genre: Femme ✗ | Ethnicite: Noir ✓
Image 3: Homme Asiatique 45   → Genre: Femme ✗ | Ethnicite: Asiatique ✓
Image 4: Femme Indienne 28    → Genre: Femme ✓ | Ethnicite: Indien ✓
Image 5: Homme Blanc 55 ans   → Genre: Homme ✓ | Ethnicite: Blanc ✓

Resultat: Genre 3/5 (60%) | Ethnicite 5/5 (100%)
Constat: Excellent pour l'ethnicite, biais vers "Femme" pour le genre
```

**Conclusion** : Les problemes de l'un sont resolus par l'autre !
- V2 Oriente → Meilleur pour le **genre**
- V4 Multitache → Meilleur pour l'**ethnicite**

**Solution : Mode Hybride** - Fusion a l'execution (runtime ensemble) :

#### Pourquoi Aucun Entrainement n'est Requis ?

Le mode Hybride utilise une technique appelee **Model Ensembling at Inference Time** (fusion de modeles a l'inference).

**Principe** : Au lieu de creer un nouveau modele, on reutilise les modeles existants deja entraines et on selectionne intelligemment leurs sorties :

```
┌─────────────────────────────────────────────────────────────────┐
│                        Image d'entree                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
          ┌─────────────────┼─────────────────┐
          ▼                 ▼                 ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│ gender_v2.tflite│ │  age_v2.tflite  │ │multitask.tflite │
│   (EXISTANT)    │ │   (EXISTANT)    │ │   (EXISTANT)    │
│   ~5.5 MB       │ │   ~6.3 MB       │ │   ~11 MB        │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         ▼                   ▼                   ▼
    Genre: 97.5%        Age: MAE 5.6      Ethnicite: 72.5%
    (on garde ✓)        (on garde ✓)       (on garde ✓)
```

**Avantages de cette approche** :
- **Pas de nouvel entrainement** : Economie de temps et de ressources GPU
- **Pas de nouveau fichier .tflite** : On reutilise les fichiers existants
- **Flexibilite** : Si un modele est ameliore, le mode Hybride en beneficie automatiquement
- **Explicabilite** : On sait exactement quel modele produit quelle prediction

**Implementation en Kotlin** :
```kotlin
// FacePredictorHybrid.kt - Charge 3 modeles, utilise le meilleur de chacun

class FacePredictorHybrid(context: Context) {
    private var genderInterpreter: Interpreter    // gender_v2_model.tflite
    private var ageInterpreter: Interpreter       // age_v2_model.tflite
    private var multitaskInterpreter: Interpreter // multitask_model.tflite

    fun predict(bitmap: Bitmap): PredictionResult {
        // Genre depuis V2 (avec CLAHE) - meilleur pour le genre
        val gender = genderInterpreter.run(bitmap)

        // Age depuis V2 - bon pour l'age
        val age = ageInterpreter.run(bitmap)

        // Ethnicite depuis V4 Multitask - meilleur pour l'ethnicite
        val ethnicity = multitaskInterpreter.run(bitmap).ethnicityOutput

        return PredictionResult(age, gender, ethnicity)
    }
}
```

**Resultat** : **97.5% sur le genre ET 72.5% sur l'ethnicite** sans aucun nouvel entrainement !

### Performances des Modeles

#### Mode Oriente V2 (3 modeles)

| Modele      | Metrique         | Valeur   |
| ----------- | ---------------- | -------- |
| Age V2      | MAE              | 7.26 ans |
| Age V2      | RMSE             | 9.63 ans |
| Gender V2   | Accuracy         | 88.2%    |
| Gender V2   | F1-Score         | 87.3%    |
| Gender V2   | AUC-ROC          | 95.6%    |
| Ethnicity V2| Accuracy         | 26.6%    |

#### Mode Multitache V4

| Tache      | Metrique   | Valeur   |
| ---------- | ---------- | -------- |
| Age        | MAE        | 6.65 ans |
| Gender     | Accuracy   | 78.1%    |
| Gender     | F1-Score   | 75.2%    |
| Ethnicity  | Accuracy   | 64.3%    |
| Ethnicity  | F1-Score   | 63.5%    |

### Structure de l'Application

```
FacePredictor/
|-- app/src/main/
|   |-- java/com/sae/facepredictor/
|   |   |-- ml/                    # Modeles TFLite
|   |   |   |-- FacePredictorModel.kt      # Multitache V4
|   |   |   |-- FacePredictorModelV2.kt    # Oriente V2
|   |   |   |-- FacePredictorHybrid.kt     # Mode Hybride (combine V2 + V4)
|   |   |   |-- FaceDetectorHelper.kt      # MediaPipe
|   |   |-- ui/                    # Interfaces
|   |   |   |-- auth/              # Login, Register
|   |   |   |-- main/              # MainActivity, ModelInfo
|   |   |   |-- camera/            # CameraActivity
|   |   |   |-- prediction/        # PredictionResult
|   |   |   |-- history/           # Historique
|   |   |-- data/                  # Firebase, Models
|   |   |-- utils/                 # SessionManager, Extensions
|   |-- assets/                    # Modeles TFLite + JSON
|   |-- res/                       # Layouts, strings, drawables
```

### Installation de l'Application

1. Ouvrir le projet `FacePredictor/` dans Android Studio
2. Synchroniser Gradle
3. Configurer Firebase (google-services.json)
4. Build > Generate Signed APK ou Run sur emulateur/device

### Configuration Firebase Requise

1. Creer un projet Firebase
2. Activer Authentication (Email + Google)
3. Activer Firestore Database
4. Telecharger `google-services.json` dans `app/`

---

## Utilisation des Modeles

### Chargement et Prediction (Python)

```python
import tensorflow as tf
from PIL import Image
import numpy as np

# Charger le modele
model = tf.keras.models.load_model('artifacts/multitask_v2_model_final.keras')

# Preparer l'image
img = Image.open('photo.jpg').convert('RGB').resize((128, 128))
img_array = np.array(img, dtype=np.float32)  # [0, 255]

# Prediction
age, gender_prob, eth_probs = model.predict(img_array[np.newaxis, ...])

# Interpretation
age_pred = age[0, 0]
gender = "Female" if gender_prob[0, 0] > 0.5 else "Male"
ethnicity = ["White", "Black", "Asian", "Indian", "Others"][np.argmax(eth_probs[0])]

print(f"Age: {age_pred:.0f} ans, Genre: {gender}, Ethnicite: {ethnicity}")
```

### Avec TFLite (Mobile)

```python
interpreter = tf.lite.Interpreter(model_path='artifacts/multitask_model.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], img_array[np.newaxis, ...])
interpreter.invoke()

age = interpreter.get_tensor(output_details[0]['index'])
gender = interpreter.get_tensor(output_details[1]['index'])
ethnicity = interpreter.get_tensor(output_details[2]['index'])
```

---

## Configuration Google Colab

```python
# 1. Verifier GPU
import tensorflow as tf
print(f"TensorFlow: {tf.__version__}")
print(f"GPU: {tf.config.list_physical_devices('GPU')}")

# 2. Monter Google Drive
from google.colab import drive
drive.mount('/content/drive')

# 3. Se positionner dans le projet
import os
os.chdir('/content/drive/MyDrive/SAE')

# 4. Telecharger UTKFace si necessaire
!pip install kaggle
!kaggle datasets download -d jangedoo/utkface-new
!unzip -q utkface-new.zip -d data/
```

---

## FAQ

**Q: Pourquoi 128x128 pixels ?**
R: Compromis entre qualite (details suffisants) et performance (entrainement rapide, modele leger).

**Q: Pourquoi pas un seul modele pour les 3 taches ?**
R: Les deux approches sont implementees ! Les modeles separes permettent l'optimisation independante, le modele multi-tache est plus compact.

**Q: Comment ameliorer les performances ?**
R: Plus de donnees, images plus grandes (224x224), modele plus gros (EfficientNetB3), ensemble de modeles.

**Q: Mon image contient le corps entier, ca marche ?**
R: Non ! Utilisez un detecteur de visage (MediaPipe, ML Kit) pour recadrer avant prediction.

**Q: Quelle version du modele utiliser ?**
R: **V2** est actuellement la meilleure. V4 est en cours de test.

---

## Livrables du Projet SAE

| Livrable                         | Statut  |
| -------------------------------- | ------- |
| Code complet (notebooks)         | Fait    |
| 3 Modeles specialises V2         | Fait    |
| Modele multi-tache V4            | Fait    |
| Transfer Learning EfficientNetB0 | Fait    |
| Modeles TFLite                   | Fait    |
| Depot GitHub                     | Fait    |
| Application Android              | Fait    |
| Detection visage (MediaPipe)     | Fait    |
| Authentification (Firebase)      | Fait    |
| Historique predictions           | Fait    |
| Switch 3 modes (Hybride/Oriente/Multitache) | Fait    |
| Rapport technique                | A faire |
| Presentation                     | A faire |

---

## Metriques d'Evaluation

| Metrique             | Applicable a     | Implementee |
| -------------------- | ---------------- | ----------- |
| Accuracy             | Genre, Ethnicite | Oui         |
| AUC-ROC              | Genre            | Oui         |
| F1-Score             | Genre, Ethnicite | Oui         |
| MAE                  | Age              | Oui         |
| MSE/RMSE             | Age              | Oui         |
| R-squared            | Age              | Oui         |
| Matrice de confusion | Genre, Ethnicite | Oui         |

---

## Contribuer

1. Forker le projet
2. Creer un environnement virtuel : `python -m venv .venv`
3. Activer : `source .venv/bin/activate`
4. Installer : `pip install -r requirements.txt`
5. Creer une branche : `git checkout -b feature/ma-feature`
6. Commiter : `git commit -m "Ajout de ma feature"`
7. Pousser : `git push origin feature/ma-feature`
8. Ouvrir une Pull Request

---

_Document cree pour le projet SAE BUT3 Informatique - Janvier 2026_
_Encadrement: Bilal Faye & Hanane Azzag (LIPN, CNRS UMR 7030, Universite Sorbonne Paris Nord)_
