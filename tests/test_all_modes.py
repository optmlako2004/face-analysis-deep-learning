#!/usr/bin/env python3
"""
Test complet des 3 modes de prédiction sur un échantillon plus large
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random
import sys
from pathlib import Path
from collections import defaultdict

# Ensure project root is importable when running from subfolders
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

import config

# Paths to models
MODEL_DIR = config.ANDROID_ASSETS_DIR
GENDER_V2_MODEL = str(MODEL_DIR / "gender_v2_model.tflite")
AGE_V2_MODEL = str(MODEL_DIR / "age_v2_model.tflite")
ETHNICITY_V2_MODEL = str(MODEL_DIR / "ethnicity_v2_model.tflite")
MULTITASK_MODEL = str(MODEL_DIR / "multitask_model.tflite")

# Labels
GENDER_LABELS = ["Homme", "Femme"]
ETHNICITY_LABELS_V2 = ["Blanc", "Noir", "Asiatique", "Indien", "Autre"]
ETHNICITY_LABELS_V4 = ["Blanc", "Noir", "Asiatique", "Indien"]

def load_interpreter(model_path):
    # Try reference kernels first; fall back to flex delegate if needed
    try:
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        )
        interpreter.allocate_tensors()
        return interpreter
    except ValueError as exc:
        if "FULLY_CONNECTED" in str(exc):
            try:
                flex_delegate = tf.lite.experimental.load_delegate("libtensorflowlite_flex.so")
                interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
                    experimental_delegates=[flex_delegate],
                )
                interpreter.allocate_tensors()
                return interpreter
            except Exception:
                msg = (
                    "FULLY_CONNECTED v12 non pris en charge par cette version de TensorFlow. "
                    "Installez TensorFlow >= 2.17 (ou tf-nightly) ou assurez-vous que libtensorflowlite_flex.so est disponible."
                )
                raise RuntimeError(msg) from exc
        raise

def preprocess_image(image_path, size=(128, 128)):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(size)
    return np.array(img, dtype=np.float32)

def apply_clahe_simple(img_array):
    luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    min_l, max_l = luminance.min(), luminance.max()
    if max_l - min_l > 0:
        scale = 255.0 / (max_l - min_l)
        return np.clip((img_array - min_l) * scale, 0, 255)
    return img_array

def get_label_from_filename(filename):
    """Parse UTKFace filename: age_gender_race_timestamp.jpg"""
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])  # 0=male, 1=female
    race = int(parts[2])    # 0=white, 1=black, 2=asian, 3=indian, 4=others

    gender_str = "Femme" if gender == 1 else "Homme"
    race_map = {0: "Blanc", 1: "Noir", 2: "Asiatique", 3: "Indien", 4: "Autre"}
    race_str = race_map.get(race, "Autre")

    return age, gender_str, race_str

def predict_oriented_v2(img_array, gender_interp, age_interp, ethnicity_interp):
    """Mode Orienté V2: 3 modèles séparés"""
    img_input = np.expand_dims(img_array, axis=0)
    img_clahe = np.expand_dims(apply_clahe_simple(img_array), axis=0)

    # Gender (with CLAHE)
    gender_interp.set_tensor(gender_interp.get_input_details()[0]['index'], img_clahe)
    gender_interp.invoke()
    gender_prob = gender_interp.get_tensor(gender_interp.get_output_details()[0]['index'])[0][0]
    gender = "Femme" if gender_prob > 0.5 else "Homme"

    # Age
    age_interp.set_tensor(age_interp.get_input_details()[0]['index'], img_input)
    age_interp.invoke()
    age = int(np.clip(age_interp.get_tensor(age_interp.get_output_details()[0]['index'])[0][0], 0, 116))

    # Ethnicity V2 (5 classes)
    ethnicity_interp.set_tensor(ethnicity_interp.get_input_details()[0]['index'], img_input)
    ethnicity_interp.invoke()
    eth_probs = ethnicity_interp.get_tensor(ethnicity_interp.get_output_details()[0]['index'])[0]
    ethnicity = ETHNICITY_LABELS_V2[np.argmax(eth_probs)]

    return gender, age, ethnicity

def predict_multitask_v4(img_array, multitask_interp):
    """Mode Multitâche V4: 1 modèle unifié"""
    img_input = np.expand_dims(img_array, axis=0)

    multitask_interp.set_tensor(multitask_interp.get_input_details()[0]['index'], img_input)
    multitask_interp.invoke()

    outputs = multitask_interp.get_output_details()
    gender_prob = multitask_interp.get_tensor(outputs[0]['index'])[0][0]
    age = int(np.clip(multitask_interp.get_tensor(outputs[1]['index'])[0][0], 0, 116))
    eth_probs = multitask_interp.get_tensor(outputs[2]['index'])[0]

    gender = "Femme" if gender_prob > 0.5 else "Homme"
    ethnicity = ETHNICITY_LABELS_V4[np.argmax(eth_probs)]

    return gender, age, ethnicity

def predict_hybrid(img_array, gender_interp, age_interp, multitask_interp):
    """Mode Hybride: Genre V2 + Age V2 + Ethnicité V4"""
    img_input = np.expand_dims(img_array, axis=0)
    img_clahe = np.expand_dims(apply_clahe_simple(img_array), axis=0)

    # Gender from V2 (with CLAHE)
    gender_interp.set_tensor(gender_interp.get_input_details()[0]['index'], img_clahe)
    gender_interp.invoke()
    gender_prob = gender_interp.get_tensor(gender_interp.get_output_details()[0]['index'])[0][0]
    gender = "Femme" if gender_prob > 0.5 else "Homme"

    # Age from V2
    age_interp.set_tensor(age_interp.get_input_details()[0]['index'], img_input)
    age_interp.invoke()
    age = int(np.clip(age_interp.get_tensor(age_interp.get_output_details()[0]['index'])[0][0], 0, 116))

    # Ethnicity from V4 Multitask
    multitask_interp.set_tensor(multitask_interp.get_input_details()[0]['index'], img_input)
    multitask_interp.invoke()
    eth_probs = multitask_interp.get_tensor(multitask_interp.get_output_details()[2]['index'])[0]
    ethnicity = ETHNICITY_LABELS_V4[np.argmax(eth_probs)]

    return gender, age, ethnicity

def main():
    print("=" * 70)
    print("TEST COMPLET DES 3 MODES DE PRÉDICTION")
    print("=" * 70)

    # Load models
    print("\nChargement des modèles...")
    gender_interp = load_interpreter(GENDER_V2_MODEL)
    age_interp = load_interpreter(AGE_V2_MODEL)
    ethnicity_interp = load_interpreter(ETHNICITY_V2_MODEL)
    multitask_interp = load_interpreter(MULTITASK_MODEL)
    print("✓ Tous les modèles chargés")

    # Get test images - sample from each category
    data_dir = config.IMAGES_DIR
    all_images = os.listdir(data_dir)

    # Filter for 4 ethnicities (exclude "Autre")
    test_images = []
    for ethnicity_id in range(4):  # 0-3 (Blanc, Noir, Asiatique, Indien)
        for gender_id in range(2):  # 0-1 (Homme, Femme)
            pattern = f"_{gender_id}_{ethnicity_id}_"
            matching = [f for f in all_images if pattern in f]
            if matching:
                # Sample up to 5 images per category
                sampled = random.sample(matching, min(5, len(matching)))
                test_images.extend(sampled)

    print(f"\nTest sur {len(test_images)} images (échantillon équilibré)")

    # Results storage
    results = {
        'oriented': {'gender_correct': 0, 'ethnicity_correct': 0, 'age_errors': []},
        'multitask': {'gender_correct': 0, 'ethnicity_correct': 0, 'age_errors': []},
        'hybrid': {'gender_correct': 0, 'ethnicity_correct': 0, 'age_errors': []}
    }

    n_valid = 0

    for img_file in test_images:
        img_path = data_dir / img_file

        try:
            true_age, true_gender, true_ethnicity = get_label_from_filename(img_file)

            # Skip "Autre" ethnicity
            if true_ethnicity == "Autre":
                continue

            img_array = preprocess_image(img_path)
            n_valid += 1

            # Oriented V2
            g_o, a_o, e_o = predict_oriented_v2(img_array, gender_interp, age_interp, ethnicity_interp)
            if g_o == true_gender: results['oriented']['gender_correct'] += 1
            if e_o == true_ethnicity: results['oriented']['ethnicity_correct'] += 1
            results['oriented']['age_errors'].append(abs(a_o - true_age))

            # Multitask V4
            g_m, a_m, e_m = predict_multitask_v4(img_array, multitask_interp)
            if g_m == true_gender: results['multitask']['gender_correct'] += 1
            if e_m == true_ethnicity: results['multitask']['ethnicity_correct'] += 1
            results['multitask']['age_errors'].append(abs(a_m - true_age))

            # Hybrid
            g_h, a_h, e_h = predict_hybrid(img_array, gender_interp, age_interp, multitask_interp)
            if g_h == true_gender: results['hybrid']['gender_correct'] += 1
            if e_h == true_ethnicity: results['hybrid']['ethnicity_correct'] += 1
            results['hybrid']['age_errors'].append(abs(a_h - true_age))

        except Exception as e:
            continue

    # Print results
    print("\n" + "=" * 70)
    print("RÉSULTATS COMPARATIFS")
    print("=" * 70)

    for mode, data in results.items():
        gender_acc = data['gender_correct'] / n_valid * 100
        eth_acc = data['ethnicity_correct'] / n_valid * 100
        age_mae = np.mean(data['age_errors'])

        mode_label = {
            'oriented': 'ORIENTÉ V2 (3 modèles)',
            'multitask': 'MULTITÂCHE V4 (1 modèle)',
            'hybrid': 'HYBRIDE (Genre V2 + Eth V4 + Age V2)'
        }[mode]

        print(f"\n{mode_label}:")
        print(f"  Genre:     {data['gender_correct']}/{n_valid} ({gender_acc:.1f}%)")
        print(f"  Ethnicité: {data['ethnicity_correct']}/{n_valid} ({eth_acc:.1f}%)")
        print(f"  Age MAE:   {age_mae:.2f} ans")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)

    # Compare
    best_gender = max(results.items(), key=lambda x: x[1]['gender_correct'])[0]
    best_eth = max(results.items(), key=lambda x: x[1]['ethnicity_correct'])[0]
    best_age = min(results.items(), key=lambda x: np.mean(x[1]['age_errors']))[0]

    print(f"Meilleur pour le genre:     {best_gender.upper()}")
    print(f"Meilleur pour l'ethnicité:  {best_eth.upper()}")
    print(f"Meilleur pour l'âge:        {best_age.upper()}")

if __name__ == "__main__":
    random.seed(42)
    main()
