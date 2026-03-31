#!/usr/bin/env python3
"""
Test comparatif des 4 modes: Oriente V2, Multitache V4, Hybride, MoE
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random
import sys
import time
from pathlib import Path

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
MOE_MODEL = str(MODEL_DIR / "moe_mobilenetv3.tflite")

# Labels
GENDER_LABELS = ["Homme", "Femme"]
ETHNICITY_LABELS_V2 = ["Blanc", "Noir", "Asiatique", "Indien", "Autre"]
ETHNICITY_LABELS_V4 = ["Blanc", "Noir", "Asiatique", "Indien"]

def load_interpreter(model_path):
    # Use builtin_ref resolver to support newer TFLite ops with older TF wheels
    try:
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF,
        )
        interpreter.allocate_tensors()
        return interpreter
    except ValueError as exc:
        # Try with Flex delegate for newer/bigger op versions (CAST v5, FULLY_CONNECTED v12, etc.)
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
                "Op non supporte par le runtime TFLite actuel. "
                "Installez TensorFlow >= 2.17 ou ajoutez libtensorflowlite_flex.so au LD_LIBRARY_PATH."
            )
            raise RuntimeError(msg) from exc

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
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])
    race = int(parts[2])
    gender_str = "Femme" if gender == 1 else "Homme"
    race_map = {0: "Blanc", 1: "Noir", 2: "Asiatique", 3: "Indien", 4: "Autre"}
    race_str = race_map.get(race, "Autre")
    return age, gender_str, race_str

def predict_oriented_v2(img_array, gender_interp, age_interp, ethnicity_interp):
    img_input = np.expand_dims(img_array, axis=0)
    img_clahe = np.expand_dims(apply_clahe_simple(img_array), axis=0)

    gender_interp.set_tensor(gender_interp.get_input_details()[0]['index'], img_clahe)
    gender_interp.invoke()
    gender_prob = gender_interp.get_tensor(gender_interp.get_output_details()[0]['index'])[0][0]
    gender = "Femme" if gender_prob > 0.5 else "Homme"

    age_interp.set_tensor(age_interp.get_input_details()[0]['index'], img_input)
    age_interp.invoke()
    age = int(np.clip(age_interp.get_tensor(age_interp.get_output_details()[0]['index'])[0][0], 0, 116))

    ethnicity_interp.set_tensor(ethnicity_interp.get_input_details()[0]['index'], img_input)
    ethnicity_interp.invoke()
    eth_probs = ethnicity_interp.get_tensor(ethnicity_interp.get_output_details()[0]['index'])[0]
    ethnicity = ETHNICITY_LABELS_V2[np.argmax(eth_probs)]

    return gender, age, ethnicity

def predict_multitask_v4(img_array, multitask_interp):
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
    img_input = np.expand_dims(img_array, axis=0)
    img_clahe = np.expand_dims(apply_clahe_simple(img_array), axis=0)

    # Gender from V2 (with CLAHE)
    gender_interp.set_tensor(gender_interp.get_input_details()[0]['index'], img_clahe)
    gender_interp.invoke()
    gender_prob = gender_interp.get_tensor(gender_interp.get_output_details()[0]['index'])[0][0]
    gender = "Femme" if gender_prob > 0.5 else "Homme"

    # Age: ensemble V2 + V4 with TTA (flip)
    img_flipped = np.expand_dims(np.flip(img_array, axis=1).copy(), axis=0)
    img_clahe_flipped = np.expand_dims(np.flip(apply_clahe_simple(img_array), axis=1).copy(), axis=0)

    age_interp.set_tensor(age_interp.get_input_details()[0]['index'], img_input)
    age_interp.invoke()
    age_v2 = age_interp.get_tensor(age_interp.get_output_details()[0]['index'])[0][0]

    age_interp.set_tensor(age_interp.get_input_details()[0]['index'], img_flipped)
    age_interp.invoke()
    age_v2_flip = age_interp.get_tensor(age_interp.get_output_details()[0]['index'])[0][0]

    multitask_interp.set_tensor(multitask_interp.get_input_details()[0]['index'], img_input)
    multitask_interp.invoke()
    age_v4 = multitask_interp.get_tensor(multitask_interp.get_output_details()[1]['index'])[0][0]
    eth_probs = multitask_interp.get_tensor(multitask_interp.get_output_details()[2]['index'])[0]

    multitask_interp.set_tensor(multitask_interp.get_input_details()[0]['index'], img_flipped)
    multitask_interp.invoke()
    age_v4_flip = multitask_interp.get_tensor(multitask_interp.get_output_details()[1]['index'])[0][0]
    eth_probs_flip = multitask_interp.get_tensor(multitask_interp.get_output_details()[2]['index'])[0]

    age = int(np.clip(np.mean([age_v2, age_v2_flip, age_v4, age_v4_flip]), 0, 116))

    # Ethnicity: ensemble V4 + TTA
    eth_avg = (eth_probs + eth_probs_flip) / 2
    ethnicity = ETHNICITY_LABELS_V4[np.argmax(eth_avg)]

    return gender, age, ethnicity

def predict_moe(img_array, moe_interp):
    img_input = np.expand_dims(img_array, axis=0)
    moe_interp.set_tensor(moe_interp.get_input_details()[0]['index'], img_input)
    moe_interp.invoke()

    outputs = moe_interp.get_output_details()
    # TFLite reorders: output[0]=age(:1), output[1]=gender(:0), output[2]=ethnicity(:2)
    age = int(np.clip(moe_interp.get_tensor(outputs[0]['index'])[0][0], 0, 116))
    gender_prob = moe_interp.get_tensor(outputs[1]['index'])[0][0]
    eth_probs = moe_interp.get_tensor(outputs[2]['index'])[0]

    gender = "Femme" if gender_prob > 0.5 else "Homme"
    ethnicity = ETHNICITY_LABELS_V4[np.argmax(eth_probs)]
    return gender, age, ethnicity

def main():
    print("=" * 70)
    print("TEST COMPARATIF DES 4 MODES DE PREDICTION")
    print("Oriente V2 | Multitache V4 | Hybride | MoE Expert")
    print("=" * 70)

    # Load models
    print("\nChargement des modeles...")
    gender_interp = load_interpreter(GENDER_V2_MODEL)
    age_interp = load_interpreter(AGE_V2_MODEL)
    ethnicity_interp = load_interpreter(ETHNICITY_V2_MODEL)
    multitask_interp = load_interpreter(MULTITASK_MODEL)
    moe_interp = load_interpreter(MOE_MODEL)

    # Log MoE output details
    print("\nMoE output details:")
    for out in moe_interp.get_output_details():
        print(f"  [{out['index']}] {out['name']}: shape={out['shape']}")

    print("Tous les modeles charges")

    # Get test images
    data_dir = config.IMAGES_DIR
    all_images = os.listdir(data_dir)

    test_images = []
    for ethnicity_id in range(4):
        for gender_id in range(2):
            pattern = f"_{gender_id}_{ethnicity_id}_"
            matching = [f for f in all_images if pattern in f]
            if matching:
                sampled = random.sample(matching, min(10, len(matching)))
                test_images.extend(sampled)

    print(f"\nTest sur {len(test_images)} images (echantillon equilibre)")

    # Results storage
    results = {
        'oriented': {'gender_correct': 0, 'ethnicity_correct': 0, 'age_errors': [], 'time': 0},
        'multitask': {'gender_correct': 0, 'ethnicity_correct': 0, 'age_errors': [], 'time': 0},
        'hybrid': {'gender_correct': 0, 'ethnicity_correct': 0, 'age_errors': [], 'time': 0},
        'moe': {'gender_correct': 0, 'ethnicity_correct': 0, 'age_errors': [], 'time': 0}
    }

    # Per-category results for MoE
    moe_gender_by_true = {'Homme': {'correct': 0, 'total': 0}, 'Femme': {'correct': 0, 'total': 0}}
    moe_eth_by_true = {e: {'correct': 0, 'total': 0} for e in ETHNICITY_LABELS_V4}

    n_valid = 0

    for img_file in test_images:
        img_path = data_dir / img_file

        try:
            true_age, true_gender, true_ethnicity = get_label_from_filename(img_file)

            if true_ethnicity == "Autre":
                continue

            img_array = preprocess_image(img_path)
            n_valid += 1

            # Oriented V2
            t0 = time.time()
            g_o, a_o, e_o = predict_oriented_v2(img_array, gender_interp, age_interp, ethnicity_interp)
            results['oriented']['time'] += time.time() - t0
            if g_o == true_gender: results['oriented']['gender_correct'] += 1
            if e_o == true_ethnicity: results['oriented']['ethnicity_correct'] += 1
            results['oriented']['age_errors'].append(abs(a_o - true_age))

            # Multitask V4
            t0 = time.time()
            g_m, a_m, e_m = predict_multitask_v4(img_array, multitask_interp)
            results['multitask']['time'] += time.time() - t0
            if g_m == true_gender: results['multitask']['gender_correct'] += 1
            if e_m == true_ethnicity: results['multitask']['ethnicity_correct'] += 1
            results['multitask']['age_errors'].append(abs(a_m - true_age))

            # Hybrid
            t0 = time.time()
            g_h, a_h, e_h = predict_hybrid(img_array, gender_interp, age_interp, multitask_interp)
            results['hybrid']['time'] += time.time() - t0
            if g_h == true_gender: results['hybrid']['gender_correct'] += 1
            if e_h == true_ethnicity: results['hybrid']['ethnicity_correct'] += 1
            results['hybrid']['age_errors'].append(abs(a_h - true_age))

            # MoE
            t0 = time.time()
            g_moe, a_moe, e_moe = predict_moe(img_array, moe_interp)
            results['moe']['time'] += time.time() - t0
            if g_moe == true_gender: results['moe']['gender_correct'] += 1
            if e_moe == true_ethnicity: results['moe']['ethnicity_correct'] += 1
            results['moe']['age_errors'].append(abs(a_moe - true_age))

            # MoE detailed tracking
            moe_gender_by_true[true_gender]['total'] += 1
            if g_moe == true_gender:
                moe_gender_by_true[true_gender]['correct'] += 1

            if true_ethnicity in moe_eth_by_true:
                moe_eth_by_true[true_ethnicity]['total'] += 1
                if e_moe == true_ethnicity:
                    moe_eth_by_true[true_ethnicity]['correct'] += 1

        except Exception as e:
            continue

    # Print results
    print("\n" + "=" * 70)
    print("RESULTATS COMPARATIFS")
    print("=" * 70)

    mode_labels = {
        'oriented': 'ORIENTE V2 (3 modeles EfficientNet)',
        'multitask': 'MULTITACHE V4 (1 modele EfficientNet)',
        'hybrid': 'HYBRIDE (Genre V2 + Ensemble Age + Eth V4 TTA)',
        'moe': 'MoE EXPERT (MobileNetV3 + 9 experts)'
    }

    for mode, data in results.items():
        gender_acc = data['gender_correct'] / n_valid * 100
        eth_acc = data['ethnicity_correct'] / n_valid * 100
        age_mae = np.mean(data['age_errors'])
        avg_time = data['time'] / n_valid * 1000  # ms

        print(f"\n{mode_labels[mode]}:")
        print(f"  Genre:     {data['gender_correct']}/{n_valid} ({gender_acc:.1f}%)")
        print(f"  Ethnicite: {data['ethnicity_correct']}/{n_valid} ({eth_acc:.1f}%)")
        print(f"  Age MAE:   {age_mae:.2f} ans")
        print(f"  Temps moy: {avg_time:.2f} ms/image")

    # MoE detailed results
    print("\n" + "=" * 70)
    print("DETAIL MoE PAR CATEGORIE")
    print("=" * 70)

    print("\nGenre:")
    for g, data in moe_gender_by_true.items():
        if data['total'] > 0:
            acc = data['correct'] / data['total'] * 100
            print(f"  {g}: {data['correct']}/{data['total']} ({acc:.1f}%)")

    print("\nEthnicite:")
    for e, data in moe_eth_by_true.items():
        if data['total'] > 0:
            acc = data['correct'] / data['total'] * 100
            print(f"  {e}: {data['correct']}/{data['total']} ({acc:.1f}%)")

    # Summary table
    print("\n" + "=" * 70)
    print("TABLEAU RECAPITULATIF")
    print("=" * 70)
    print(f"{'Mode':<35} {'Genre%':>8} {'Ethnie%':>8} {'Age MAE':>8} {'Temps ms':>9}")
    print("-" * 70)
    for mode, data in results.items():
        gender_acc = data['gender_correct'] / n_valid * 100
        eth_acc = data['ethnicity_correct'] / n_valid * 100
        age_mae = np.mean(data['age_errors'])
        avg_time = data['time'] / n_valid * 1000
        print(f"{mode_labels[mode][:35]:<35} {gender_acc:>7.1f}% {eth_acc:>7.1f}% {age_mae:>7.2f} {avg_time:>8.2f}")

    # Best for each metric
    print("\n" + "=" * 70)
    print("MEILLEUR PAR METRIQUE")
    print("=" * 70)
    best_gender = max(results.items(), key=lambda x: x[1]['gender_correct'])[0]
    best_eth = max(results.items(), key=lambda x: x[1]['ethnicity_correct'])[0]
    best_age = min(results.items(), key=lambda x: np.mean(x[1]['age_errors']))[0]
    best_speed = min(results.items(), key=lambda x: x[1]['time'])[0]

    print(f"Meilleur genre:     {best_gender.upper()}")
    print(f"Meilleur ethnicite: {best_eth.upper()}")
    print(f"Meilleur age:       {best_age.upper()}")
    print(f"Plus rapide:        {best_speed.upper()}")

if __name__ == "__main__":
    random.seed(42)
    main()
