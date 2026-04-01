# Rapport de projet SAE — FacePredictor

## Application mobile intelligente pour la prediction d'age, de genre et d'ethnicite a partir du visage

SAE BUT 3 Informatique — Annee 2025-2026

Universite Sorbonne Paris Nord — LIPN, CNRS UMR 7030

Encadrement : Bilal Faye

Equipe : Ako Christian, Calrd Similien, Noe Cervera, Dhanoush Kessavane

Mars 2026

---

# 1. Introduction

Ce rapport presente le travail realise dans le cadre de la SAE de BUT 3 Informatique. L'objectif est de developper une application mobile Android capable de predire l'age, le genre et l'ethnicite d'une personne a partir d'une photo de son visage, en s'appuyant sur des modeles de Deep Learning entraines sur le dataset UTKFace et deployes avec TensorFlow Lite.

Le projet couvre l'integralite de la chaine de production : de la collecte et l'analyse des donnees, en passant par l'entrainement et la comparaison de plusieurs architectures de reseaux de neurones, jusqu'au deploiement dans une application Android fonctionnelle avec authentification, historique et mode temps reel.


# 2. Analyse du sujet

Le sujet demande d'implementer trois strategies de modelisation distinctes. La premiere consiste a entrainer trois modeles specialises, un pour chaque tache (age, genre, ethnicite). La deuxieme demande un modele multi-tache capable de predire les trois attributs simultanement. La troisieme impose de comparer plusieurs architectures de reseaux de neurones par transfert de connaissances, en evaluant MobileNetV2, EfficientNetB0, ResNet50 et MobileNetV3Large.

L'application doit egalement proposer la capture d'image via la camera, le chargement depuis la galerie, une authentification, un historique des predictions, les informations sur le modele utilise, et en bonus un mode temps reel.

Toutes les metriques suivantes doivent etre calculees : Accuracy, AUC, AP, MAE, MSE, R carre, ARI et NMI.

*Repartition du travail*

Ako Christian a pris en charge l'architecture Machine Learning, l'entrainement des modeles et l'integration TFLite. Calrd Similien s'est occupe du developpement Android, de l'integration et de la navigation. Noe Cervera a travaille sur l'entrainement, l'evaluation des metriques et les notebooks. Dhanoush Kessavane a gere le front-end, le design et la documentation.


# 3. Donnees et preprocessing

## Le dataset UTKFace

Le dataset UTKFace contient 23 708 images de visages recadres, annotees avec trois labels : l'age (valeur continue de 0 a 116 ans), le genre (homme ou femme) et l'ethnicite (cinq classes : White, Black, Asian, Indian et Other). Le format de nommage des fichiers est age_gender_race_date.jpg.

## Desequilibre du dataset

Le dataset presente un desequilibre notable pour l'ethnicite. La classe White represente environ 45 pourcent des images, tandis que la classe Other ne represente que 5 pourcent. Pour compenser ce desequilibre, nous utilisons la Focal Loss pour l'ethnicite, qui penalise moins les exemples bien classes et se concentre sur les exemples difficiles. Les poids de classe sont calcules de maniere inversement proportionnelle a la frequence de chaque classe.

## Split et augmentation

Les donnees sont divisees en 70 pourcent pour l'entrainement, 15 pourcent pour la validation et 15 pourcent pour le test. Le split est stratifie sur l'ethnicite pour maintenir les proportions dans chaque ensemble.

Le preprocessing comprend le redimensionnement a 224 par 224 pixels, taille d'entree standard des backbones pre-entraines sur ImageNet. Pendant l'entrainement, nous appliquons de la data augmentation : flip horizontal aleatoire, crop aleatoire, variations de luminosite, de contraste et de saturation. L'augmentation n'est pas appliquee sur les ensembles de validation et de test.

Nous utilisons egalement la mixed precision float16 pour accelerer l'entrainement sur notre GPU NVIDIA RTX 4050 Laptop disposant de 6 Go de VRAM.


# 4. Detection de visage — MediaPipe

Nos modeles ont ete entraines sur UTKFace, un dataset de visages deja recadres. En production, l'utilisateur prend une photo qui peut contenir le corps entier, un arriere-plan complexe ou plusieurs personnes. Il est donc indispensable de detecter et isoler le visage avant de l'envoyer au modele de prediction.

Nous utilisons MediaPipe Face Detection, un modele pre-entraine de Google optimise pour le mobile. Il detecte les visages en 15 a 30 millisecondes et fonctionne meme a distance. L'image du visage detecte est recadree en carre avec un padding de 30 pourcent autour du visage pour inclure le contour.

En mode temps reel, un guide ovale est affiche pour aider l'utilisateur a positionner son visage. Lorsqu'un visage est detecte, le guide disparait et un rectangle vert entoure le visage.

Sans cette etape, le modele analyserait les vetements, les cheveux ou l'arriere-plan au lieu du visage, ce qui donnerait des predictions incorrectes.


# 5. Strategie 1 — Trois modeles specialises

## Architecture

Chaque modele est compose d'un backbone EfficientNetB0 pre-entraine sur ImageNet, suivi d'un GlobalAveragePooling2D, d'une couche BatchNormalization, d'un Dropout a 0.3 et de couches Dense specifiques a la tache.

Le modele d'age utilise une sortie Dense avec activation lineaire pour la regression. Le modele de genre utilise une sortie Dense avec activation sigmoid pour la classification binaire. Le modele d'ethnicite utilise une sortie Dense a 5 neurones avec activation softmax pour la classification multi-classe.

## Entrainement en deux phases

La premiere phase, le warmup, dure 15 epochs avec le backbone entierement gele. Seules les couches Dense apprennent avec un learning rate de 1e-3. Un EarlyStopping avec patience de 7 surveille la validation loss.

La deuxieme phase, le fine-tuning, dure jusqu'a 40 epochs. Les 15 dernieres couches du backbone sont debloquees, tandis que les couches BatchNormalization restent gelees pour eviter le catastrophic forgetting. Le learning rate est reduit a 1e-5 et le batch size a 4 en raison des contraintes de VRAM.

## Fonctions de perte

Le modele d'age utilise la Huber Loss avec un delta de 5, robuste aux outliers que representent les ages extremes. Le modele de genre utilise la Binary Cross-Entropy avec un label smoothing de 0.05. Le modele d'ethnicite utilise la Focal Loss avec gamma de 2 et des poids de classe pour gerer le desequilibre.

## Resultats

Le modele d'age atteint un MAE d'environ 6.2 ans. Le modele de genre atteint environ 90 pourcent d'accuracy. Le modele d'ethnicite atteint environ 70 pourcent d'accuracy.


# 6. Strategie 2 — Modele multi-tache Mixture of Experts

## Initiative personnelle

Le sujet demandait un modele multi-tache simple. Apres nos recherches, nous avons voulu aller plus loin en testant une architecture Mixture of Experts (MoE), qui ameliore le modele multi-tache classique en ajoutant des experts specialises.

## Architecture MoE

Le backbone EfficientNetB0 est partage entre les trois taches. Apres le GlobalAveragePooling, BatchNormalization et Dropout, trois tetes MoE independantes predisent respectivement l'age, le genre et l'ethnicite.

Chaque tete MoE contient quatre experts, qui sont des petits MLP independants composes d'une couche Dense a 256 neurones avec activation ReLU, un Dropout a 0.2, puis une couche Dense de sortie. Un gating network, compose d'une couche Dense a 128 neurones suivie d'un softmax a 4 sorties, determine dynamiquement le poids de chaque expert pour chaque image.

Le principe est que chaque expert se specialise sur un sous-ensemble de visages. Par exemple, un expert peut apprendre a predire l'age des enfants, un autre celui des personnes agees. Le gating network apprend a router chaque image vers l'expert le plus adapte.

## Entrainement en trois phases

La premiere phase, le warmup, dure 25 epochs avec le backbone entierement gele et un learning rate de 1e-3. Les poids des pertes sont ajustes pour privilegier le genre et l'ethnicite : age a 0.3, genre a 1.5 et ethnicite a 2.0.

La deuxieme phase, le fine-tune partiel, dure 20 epochs. Les 15 dernieres couches du backbone sont debloquees avec un learning rate de 2e-5 et un batch size de 4.

La troisieme phase, le fine-tune etendu, dure 15 epochs. 30 couches sont debloquees avec un learning rate de 5e-6. Entre chaque phase, un nettoyage complet de la memoire GPU est effectue.

## Resultats

Le modele multi-tache MoE atteint 90.7 pourcent d'accuracy en genre avec un AUC de 96.4 pourcent. En ethnicite, il atteint 70.8 pourcent d'accuracy avec un AUC de 90.6 pourcent. Pour l'age, le MAE est de 6.36 ans avec un R carre de 0.778.

Les autres metriques demandees sont : AP de 95.3 pourcent pour le genre et 80.8 pourcent pour l'ethnicite, ARI de 0.663 pour le genre et 0.452 pour l'ethnicite, NMI de 0.554 pour le genre et 0.408 pour l'ethnicite, et MSE de 86.3 pour l'age.


# 7. Mode Hybride — Initiative de l'equipe

## L'idee

Apres avoir entraine les trois modeles specialises et le modele multi-tache MoE, nous avons constate que chaque modele avait ses forces : le modele d'age specialise avait le meilleur MAE, le modele de genre la meilleure accuracy, etc. Nous avons alors eu l'idee de combiner les meilleurs points de chaque modele dans un mode unique que nous avons appele Hybride.

## Fonctionnement

Le mode Hybride charge les trois modeles specialises (age, genre, ethnicite) et applique du Test Time Augmentation (TTA). Pour chaque image, le mode effectue deux inferences par modele : une sur l'image originale et une sur l'image retournee horizontalement. Les predictions sont ensuite moyennees, ce qui reduit le bruit et stabilise les resultats.

## Resultats

Le mode Hybride obtient les meilleurs resultats de tous nos modes. Sur un echantillon de 10 images du jeu de test, il atteint 100 pourcent de precision en genre (10/10), 80 pourcent en ethnicite (8/10) et un MAE de 3.3 ans en age. Ces resultats sont superieurs au MoE (8/10 genre, 6/10 ethnicite, 3.9 MAE) et aux modeles individuels sans TTA.

C'est pour cette raison que le mode Hybride est le mode par defaut de l'application.


# 8. Strategie 3 — Comparaison des backbones

## Protocole

Nous avons compare quatre architectures pre-entrainees sur ImageNet : MobileNetV2 (3.4 millions de parametres), EfficientNetB0 (5.3 millions), ResNet50 (25.6 millions) et MobileNetV3Large (4.2 millions).

Chaque backbone est entraine avec la meme architecture (backbone partage plus trois tetes MoE) pendant 8 epochs de warmup puis 7 epochs de fine-tune. Un score composite est calcule en ponderant les metriques, avec le genre et l'ethnicite ponderes trois fois plus que l'age.

## Resultats de la comparaison

EfficientNetB0 obtient le meilleur score composite de 9.17 avec 90.1 pourcent d'accuracy en genre, 68.3 pourcent en ethnicite et un MAE de 6.19 en age, pour un temps d'entrainement de 15 minutes.

ResNet50 obtient un score tres proche de 9.15 avec le meilleur MAE age de 5.87, mais il est cinq fois plus gros (30 millions de parametres) et deux fois plus lent (33 minutes).

MobileNetV3Large obtient un score de 9.02 et est le plus rapide (11 minutes), mais legerement moins precis.

MobileNetV2 est le moins performant avec un score de 8.37, surtout en ethnicite ou il n'atteint que 55.7 pourcent.

## Choix du backbone

EfficientNetB0 est selectionne comme meilleur backbone. Il offre le meilleur compromis entre precision, taille du modele et vitesse d'entrainement. Sa taille raisonnable (environ 8 Mo en TFLite) le rend adapte au deploiement mobile.


# 9. Interpretabilite du modele

## Pourquoi l'interpretabilite

Il est important de comprendre comment le modele prend ses decisions, pour verifier qu'il utilise des caracteristiques pertinentes, identifier les biais potentiels et pouvoir expliquer les resultats.

## Methode utilisee

Nous utilisons des saliency maps basees sur le gradient de la sortie par rapport aux pixels d'entree. Les zones avec un gradient eleve sont celles que le modele considere comme decisives pour sa prediction. Ces cartes de chaleur sont generees pour chaque tache (age, genre, ethnicite) sur les modeles multitache et individuels.

## Ce que le modele regarde

Pour l'age, le modele se concentre sur les rides, la texture de la peau, la couleur des cheveux et les proportions du visage. Pour le genre, il regarde la forme de la machoire, la pilosite faciale et la structure osseuse. Pour l'ethnicite, il analyse le teint de la peau, la forme des yeux et du nez, et les proportions generales du visage.

Le modele se focalise principalement sur le visage et non sur les cheveux, les vetements ou l'arriere-plan, ce qui confirme qu'il a appris des features biologiquement coherentes.

Le modele n'utilise ni infrarouge ni capteur thermique. Il analyse uniquement les patterns statistiques dans les pixels de l'image.


# 10. Deploiement TensorFlow Lite

## Conversion

Les modeles Keras sont convertis en TensorFlow Lite avec optimisation par quantification dynamique et types float16. Les operations TFLITE_BUILTINS et SELECT_TF_OPS sont utilisees pour supporter les couches MoE.

## Modeles deployes dans l'application

Tous les modeles ont ete re-entraines avec EfficientNetB0 comme backbone (224x224), selectionne grace a la comparaison du notebook 1.

Les trois modeles individuels (age, genre, ethnicite) pesent chacun 8.4 Mo en TFLite. Le modele multitache MoE pese 17 Mo. Le modele de detection de visage MediaPipe pese 0.2 Mo.

Le temps moyen d'inference est d'environ 100 a 200 millisecondes par image sur un smartphone de milieu de gamme.


# 11. Application Android

## Technologies utilisees

L'application est developpee en Kotlin avec le SDK Android 34 (minimum 24, soit Android 7.0 et plus). L'inference des modeles utilise TensorFlow Lite 2.17.0. La detection de visage repose sur MediaPipe 0.10.9. L'authentification et le stockage sont geres par Firebase Auth et Firestore (BoM 33.7.0). La capture photo et le mode temps reel utilisent CameraX 1.3.1. L'interface suit les guidelines Material Design 3.

## Modes de prediction

L'application propose trois modes de prediction selectionnables dans les parametres.

Le mode Hybride, recommande par defaut, utilise les trois modeles specialises avec du Test Time Augmentation (flip horizontal et moyenne des predictions) pour des resultats plus stables.

Le mode 3 Modeles specialises utilise les trois modeles EfficientNetB0 individuels, un par tache.

Le mode Multitache MoE utilise le modele Mixture of Experts entraine lors de ce projet, qui predit les trois taches simultanement grace a des experts specialises.

## Fonctionnalites implementees

L'utilisateur peut capturer une image via la camera frontale ou arriere, charger une photo depuis la galerie, ou utiliser le mode temps reel qui analyse en continu via la camera.

L'authentification supporte la connexion par email et mot de passe ainsi que Google Sign-In via Firebase. L'historique des predictions est stocke dans Firestore avec des filtres par genre, age et ethnicite.

La gestion de compte comprend un profil avec avatar et initiales, les statistiques d'utilisation (nombre d'analyses, age moyen, genre le plus frequent), les informations de securite (membre depuis, derniere connexion, email verifie, methodes d'authentification), la modification du nom, l'export des donnees au format JSON, et la suppression du compte.

Un mode sombre complet est disponible avec un toggle dans les parametres. Un tutoriel interactif accueille les nouveaux utilisateurs lors de leur premiere connexion.

## Avantages et inconvenients de chaque mode

Le mode Hybride offre la meilleure precision grace au TTA mais charge trois modeles en memoire, ce qui le rend plus lent. Le mode 3 Modeles specialises est rapide et chaque modele est optimise pour sa tache, mais il n'y a pas de partage de features entre les taches. Le mode Multitache MoE utilise un seul fichier avec des experts specialises par type de visage, mais necessite la librairie supplementaire select-tf-ops ce qui alourdit l'APK.

## Design

L'interface suit une palette "Tech bleu / gris" professionnelle. La couleur primaire est un bleu navy, la secondaire un steel blue. Le design utilise des panels avec effets glassmorphiques, des badges, et supporte entierement le mode sombre.


# 12. Resultats et metriques

## Metriques d'evaluation

Toutes les metriques demandees par le sujet sont calculees. Pour la classification (genre et ethnicite), nous mesurons l'Accuracy, l'AUC, l'AP, l'ARI et le NMI. Pour la regression (age), nous mesurons le MAE, le MSE et le R carre.

## Synthese des resultats

Le modele multitache MoE atteint 90.7 pourcent d'accuracy en genre, 70.8 pourcent en ethnicite et 6.36 ans de MAE en age.

Les trois modeles individuels atteignent des performances comparables : environ 90 pourcent en genre, 70 pourcent en ethnicite et 6.2 ans de MAE en age.

Le mode Hybride (3 modeles + TTA) teste sur 20 images atteint 95 pourcent en genre, 90 pourcent en ethnicite et 3.2 ans de MAE en age.

## Confiance des predictions

La confiance du genre provient directement de la probabilite sigmoid du modele. La confiance de l'ethnicite provient de la probabilite softmax maximale. La confiance de l'age est estimee en fonction de la representativite de la tranche d'age dans le dataset : les ages de 15 a 55 ans, mieux representes, obtiennent une confiance plus elevee.


# 13. Difficultes rencontrees

La principale contrainte a ete la VRAM limitee a 6 Go de notre GPU NVIDIA RTX 4050 Laptop. Cela a impose un batch size reduit a 4 pendant le fine-tuning, un nombre de couches debloquees limite a 15 puis 30 au lieu de tout le backbone, un nettoyage memoire systematique entre les phases d'entrainement, et une limite explicite de la VRAM a 5 Go pour eviter les crashs du systeme.

Le desequilibre du dataset a ete compense par la Focal Loss et les poids de classe, mais l'accuracy sur la classe Other reste inferieure, autour de 50 pourcent.

Le modele confond parfois les classes Asian et Indian en raison de traits visuels proches. La matrice de confusion montre qu'environ 15 pourcent des Indians sont predits comme Asian et inversement.

Les premieres executions donnaient des resultats variables entre les backbones. La fixation des seeds pour numpy, tensorflow et random a resolu ce probleme de reproductibilite.


# 14. Conclusion

Nous avons implemente avec succes les trois strategies de modelisation demandees par le sujet et deploye le meilleur modele dans une application Android fonctionnelle et professionnelle. Le MoE, ajoute par initiative personnelle apres nos recherches, a permis d'ameliorer les performances du modele multi-tache.

Les performances obtenues sont satisfaisantes : 90.7 pourcent de precision sur le genre, 70.8 pourcent sur l'ethnicite, et 6.36 ans d'erreur moyenne sur l'age.

L'application offre une experience utilisateur complete avec trois modes de prediction, authentification Firebase, historique, mode temps reel, mode sombre, et un design moderne.

En perspectives, un dataset plus grand et mieux equilibre permettrait d'ameliorer les performances sur les classes sous-representees. Des architectures plus recentes comme EfficientNetV2, ConvNeXt ou les Vision Transformers pourraient etre testees. La quantification int8 reduirait la taille des modeles pour un deploiement plus leger.