# FacePredictor

## Prediction d'age, de genre et d'ethnicite par Deep Learning

SAE BUT 3 Informatique — 2025-2026

Ako Christian, Calrd Similien, Noe Cervera, Dhanoush Kessavane

Encadrement : Bilal Faye et Hanane Azzag — LIPN, CNRS UMR 7030


# Sommaire

1. Objectif du projet
2. Dataset UTKFace
3. Les 3 strategies de modelisation
4. Comparaison des backbones
5. Interpretabilite du modele
6. Application Android
7. Resultats finaux
8. Difficultes et solutions
9. Conclusion


# Objectif du projet

Developper une application Android capable de predire l'age, le genre et l'ethnicite d'une personne a partir d'une photo de son visage grace au Deep Learning.

*Stack technique*

- TensorFlow et Keras pour l'entrainement des modeles
- TensorFlow Lite pour le deploiement mobile
- MediaPipe pour la detection de visage
- Firebase pour l'authentification et le stockage
- Kotlin et Android SDK pour l'application mobile


# Dataset UTKFace

23 708 images de visages annotes avec trois labels : l'age de 0 a 116 ans, le genre et l'ethnicite en cinq classes (White, Black, Asian, Indian, Other).

Le dataset est desequilibre, avec 45 pourcent de White contre 5 pourcent de Other. Pour compenser, nous utilisons la Focal Loss et des poids de classe.

*Preprocessing*

- Redimensionnement a 224 par 224 pixels
- Data augmentation : flip, crop, luminosite, contraste, saturation
- Mixed precision float16 pour accelerer l'entrainement
- Split stratifie : 70 pourcent train, 15 pourcent validation, 15 pourcent test


# Strategie 1 — Trois modeles specialises

Un modele dedie par tache, chacun avec un backbone EfficientNetB0 pre-entraine sur ImageNet.

Le modele d'age fait de la regression avec une sortie lineaire et la Huber Loss. Le modele de genre fait de la classification binaire avec sigmoid et Binary Cross-Entropy. Le modele d'ethnicite fait de la classification multi-classe avec softmax et Focal Loss.

Entrainement en deux phases : warmup avec backbone gele (15 epochs), puis fine-tuning avec 15 couches debloquees (40 epochs).


# Strategie 2 — Modele multi-tache MoE

Un seul modele predit les trois taches simultanement grace a une architecture Mixture of Experts.

Le backbone EfficientNetB0 est partage. Trois tetes MoE independantes contiennent chacune quatre experts (petits MLP) et un gating network qui determine dynamiquement le poids de chaque expert.

Chaque expert se specialise sur un sous-ensemble de visages. Le gating network apprend a router chaque image vers l'expert le plus adapte.

Entrainement en trois phases : warmup (25 epochs), fine-tune partiel avec 15 couches (20 epochs), fine-tune etendu avec 30 couches (15 epochs).


# Strategie 3 — Comparaison des backbones

Quatre architectures comparees : MobileNetV2, EfficientNetB0, ResNet50 et MobileNetV3Large.

*Resultats*

- EfficientNetB0 : 90.1 pourcent genre, 68.3 pourcent ethnicite, 6.19 MAE age — Score 9.17
- ResNet50 : 89.9 pourcent genre, 66.9 pourcent ethnicite, 5.87 MAE age — Score 9.15
- MobileNetV3Large : 88.5 pourcent genre, 67.8 pourcent ethnicite, 7.11 MAE age — Score 9.02
- MobileNetV2 : 85.0 pourcent genre, 55.7 pourcent ethnicite, 7.59 MAE age — Score 8.37

EfficientNetB0 est selectionne : meilleur compromis precision, taille et vitesse. ResNet50 est quasi egal mais cinq fois plus gros.


# Interpretabilite — Grad-CAM

Grad-CAM permet de visualiser ce que le modele regarde pour chaque prediction grace a des cartes de chaleur.

Pour l'age, le modele se concentre sur les rides, la texture de peau et les cheveux. Pour le genre, il regarde la machoire, la pilosite et la structure osseuse. Pour l'ethnicite, il analyse le teint, la forme des yeux et du nez.

Le modele n'utilise ni infrarouge ni capteur special. Il analyse uniquement les patterns statistiques dans les pixels de l'image.


# Application Android

*Fonctionnalites principales*

- Quatre modes de prediction : MoE Expert (defaut), Oriente V2, Multitache, Hybride
- Capture camera frontale et arriere, galerie, mode temps reel
- Authentification Firebase : email et mot de passe, Google Sign-In
- Historique des predictions avec filtres (Firestore)
- Gestion de compte avancee : statistiques, securite, export JSON
- Mode sombre complet
- Tutoriel interactif pour les nouveaux utilisateurs
- Detection de visage avec MediaPipe et guide ovale

*Le modele par defaut est le MoE Expert avec backbone EfficientNetB0, entraine lors de ce projet.*


# Resultats finaux

*Modele Multitache MoE (modele par defaut de l'application)*

- Genre : 90.7 pourcent d'accuracy, AUC de 96.4 pourcent
- Ethnicite : 70.8 pourcent d'accuracy, AUC de 90.6 pourcent
- Age : 6.36 ans de MAE, R carre de 0.778

Toutes les metriques du sujet sont calculees : Accuracy, AUC, AP, MAE, MSE, R carre, ARI et NMI.

Les predictions visuelles sur le jeu de test montrent que le modele predit correctement le genre et l'ethnicite dans la majorite des cas, avec une erreur d'age generalement inferieure a 10 ans.


# Difficultes et solutions

Notre GPU dispose de 6 Go de VRAM. Pour eviter les crashs, nous avons reduit le batch size a 4, limite les couches debloquees a 15 puis 30, et plafonne la VRAM a 5 Go.

Le desequilibre du dataset est compense par la Focal Loss et les poids de classe.

La confusion entre Asian et Indian est une limite du dataset, les traits visuels etant proches.

La fixation des seeds a resolu les problemes de reproductibilite entre les executions.


# Conclusion

Les trois strategies de modelisation demandees ont ete implementees avec succes. Le meilleur modele (MoE Expert, EfficientNetB0) est deploye dans une application Android fonctionnelle et professionnelle.

*Perspectives*

- Dataset plus grand et mieux equilibre
- Architectures plus recentes (EfficientNetV2, Vision Transformers)
- Quantification int8 pour un deploiement plus leger
- Detection multi-visages


# Merci de votre attention

Questions ?

Depot GitHub : github.com/optmlako2004/face-analysis-deep-learning

Ako Christian, Calrd Similien, Noe Cervera, Dhanoush Kessavane