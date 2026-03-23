#!/usr/bin/env python3
"""
Diagnostic complet des modèles TFLite - MoE vs autres modes
Objectif: identifier pourquoi le MoE donne de mauvaises prédictions dans l'app
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import random
import sys

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "FacePredictor/app/src/main/assets"
DATA_DIR = "data/UTKFace"

MOE_MODEL = os.path.join(MODEL_DIR, "moe_mobilenetv3.tflite")
MULTITASK_MODEL = os.path.join(MODEL_DIR, "multitask_model.tflite")
MULTITASK_V5_MODEL = os.path.join(MODEL_DIR, "multitask_model_v5_mobilenet.tflite")
GENDER_V2_MODEL = os.path.join(MODEL_DIR, "gender_v2_model.tflite")
AGE_V2_MODEL = os.path.join(MODEL_DIR, "age_v2_model.tflite")
ETHNICITY_V2_MODEL = os.path.join(MODEL_DIR, "ethnicity_v2_model.tflite")
GENDER_V3_MODEL = os.path.join(MODEL_DIR, "gender_v3_mobilenet.tflite")
AGE_V3_MODEL = os.path.join(MODEL_DIR, "age_v3_mobilenet.tflite")
ETHNICITY_V3_MODEL = os.path.join(MODEL_DIR, "ethnicity_v3_mobilenet.tflite")

ETHNICITY_LABELS = ["Blanc", "Noir", "Asiatique", "Indien"]
ETHNICITY_LABELS_V2 = ["Blanc", "Noir", "Asiatique", "Indien", "Autre"]

def load_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp

def inspect_model(name, path):
    """Inspecte les entrées/sorties d'un modèle TFLite"""
    print(f"\n{'='*60}")
    print(f"INSPECTION: {name}")
    print(f"Fichier: {path}")
    print(f"Taille: {os.path.getsize(path) / 1024 / 1024:.2f} MB")
    print(f"{'='*60}")

    interp = load_interpreter(path)

    print("\n--- INPUTS ---")
    for inp in interp.get_input_details():
        print(f"  [{inp['index']}] {inp['name']}")
        print(f"    Shape: {inp['shape']}, Dtype: {inp['dtype']}")
        if 'quantization' in inp:
            print(f"    Quantization: {inp['quantization']}")

    print("\n--- OUTPUTS ---")
    for out in interp.get_output_details():
        print(f"  [{out['index']}] {out['name']}")
        print(f"    Shape: {out['shape']}, Dtype: {out['dtype']}")

    return interp

def preprocess_image(image_path, size=128):
    """Charge et prépare une image (identique à l'app Android)"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size), Image.BILINEAR)
    return np.array(img, dtype=np.float32)  # [0, 255]

def get_label_from_filename(filename):
    parts = filename.split('_')
    age = int(parts[0])
    gender = int(parts[1])  # 0=male, 1=female
    race = int(parts[2])
    return age, gender, race

def test_moe_output_order(moe_interp):
    """Test critique: vérifier l'ordre des sorties MoE"""
    print("\n" + "="*60)
    print("TEST CRITIQUE: ORDRE DES SORTIES MoE")
    print("="*60)

    outputs = moe_interp.get_output_details()
    print(f"\nNombre de sorties: {len(outputs)}")
    for i, out in enumerate(outputs):
        print(f"  Output[{i}] -> index={out['index']}, name='{out['name']}', shape={out['shape']}")

    # Test avec une image connue: jeune femme blanche
    # Chercher une image: 25_1_0_xxx.jpg (25 ans, femme, blanche)
    test_files = [f for f in os.listdir(DATA_DIR) if f.startswith("25_1_0_")]
    if not test_files:
        test_files = [f for f in os.listdir(DATA_DIR) if f.startswith("25_1_")]

    if test_files:
        test_img = preprocess_image(os.path.join(DATA_DIR, test_files[0]))
        img_input = np.expand_dims(test_img, axis=0)

        moe_interp.set_tensor(moe_interp.get_input_details()[0]['index'], img_input)
        moe_interp.invoke()

        print(f"\nImage test: {test_files[0]} (attendu: 25 ans, Femme, Blanche)")
        print(f"\nValeurs brutes des sorties:")
        for i, out in enumerate(outputs):
            val = moe_interp.get_tensor(out['index'])
            print(f"  Output[{i}] ({out['name']}): {val}")

        # Interprétation selon le mapping de l'app (0=age, 1=gender, 2=ethnicity)
        age_val = moe_interp.get_tensor(outputs[0]['index'])[0][0]
        gender_val = moe_interp.get_tensor(outputs[1]['index'])[0][0]
        eth_val = moe_interp.get_tensor(outputs[2]['index'])[0]

        print(f"\nMapping APP (0=age, 1=gender, 2=eth):")
        print(f"  Age (output[0]): {age_val:.2f}")
        print(f"  Gender (output[1]): {gender_val:.4f} -> {'Femme' if gender_val > 0.5 else 'Homme'}")
        print(f"  Ethnicity (output[2]): {eth_val} -> {ETHNICITY_LABELS[np.argmax(eth_val)]}")

        # Test avec mapping inverse pour vérifier
        print(f"\nTest mapping ALTERNATIF (0=gender, 1=age, 2=eth):")
        print(f"  Gender (output[0]): {moe_interp.get_tensor(outputs[0]['index'])[0][0]:.4f}")
        print(f"  Age (output[1]): {moe_interp.get_tensor(outputs[1]['index'])[0][0]:.2f}")

        print(f"\nTest mapping ALTERNATIF2 (0=eth, 1=age, 2=gender):")
        print(f"  Ethnicity (output[0]): {moe_interp.get_tensor(outputs[0]['index'])[0]}")
        print(f"  Age (output[1]): {moe_interp.get_tensor(outputs[1]['index'])[0][0]:.2f}")
        print(f"  Gender (output[2]): {moe_interp.get_tensor(outputs[2]['index'])[0]}")

def test_moe_with_known_images(moe_interp):
    """Test MoE sur des images à labels connus, sans TTA"""
    print("\n" + "="*60)
    print("TEST MoE SUR IMAGES CONNUES (sans TTA)")
    print("="*60)

    outputs = moe_interp.get_output_details()

    # Sélectionner des cas clairs
    test_cases = [
        ("5_0_0_", "5 ans, Homme, Blanc"),
        ("25_1_0_", "25 ans, Femme, Blanche"),
        ("50_0_1_", "50 ans, Homme, Noir"),
        ("30_1_2_", "30 ans, Femme, Asiatique"),
        ("40_0_3_", "40 ans, Homme, Indien"),
        ("60_1_0_", "60 ans, Femme, Blanche"),
        ("8_0_2_", "8 ans, Homme, Asiatique"),
        ("20_1_1_", "20 ans, Femme, Noire"),
    ]

    for prefix, desc in test_cases:
        files = [f for f in os.listdir(DATA_DIR) if f.startswith(prefix)]
        if not files:
            continue

        img_path = os.path.join(DATA_DIR, files[0])
        img = preprocess_image(img_path)
        img_input = np.expand_dims(img, axis=0)

        moe_interp.set_tensor(moe_interp.get_input_details()[0]['index'], img_input)
        moe_interp.invoke()

        age_raw = moe_interp.get_tensor(outputs[0]['index'])[0][0]
        gender_raw = moe_interp.get_tensor(outputs[1]['index'])[0][0]
        eth_raw = moe_interp.get_tensor(outputs[2]['index'])[0]

        pred_age = int(np.clip(age_raw, 0, 116))
        pred_gender = "Femme" if gender_raw > 0.5 else "Homme"
        pred_eth = ETHNICITY_LABELS[np.argmax(eth_raw)]

        print(f"\n  {files[0]}")
        print(f"  Attendu:  {desc}")
        print(f"  Prédit:   Age={pred_age} (brut={age_raw:.2f}), Genre={pred_gender} (prob={gender_raw:.4f}), Eth={pred_eth} (probs={np.round(eth_raw, 3)})")

        # Vérifier si les valeurs sont dans des ranges raisonnables
        if age_raw < -10 or age_raw > 150:
            print(f"  ⚠️  AGE HORS RANGE: {age_raw}")
        if gender_raw < 0 or gender_raw > 1:
            print(f"  ⚠️  GENDER HORS [0,1]: {gender_raw}")


def test_all_models_comparison(n_samples=100):
    """Compare tous les modèles sur un échantillon UTKFace"""
    print("\n" + "="*60)
    print(f"COMPARAISON DE TOUS LES MODELES ({n_samples} images)")
    print("="*60)

    # Load all models
    models = {}
    model_files = {
        'MoE': MOE_MODEL,
        'Multitask_V4': MULTITASK_MODEL,
        'Multitask_V5': MULTITASK_V5_MODEL,
        'Gender_V2': GENDER_V2_MODEL,
        'Age_V2': AGE_V2_MODEL,
        'Ethnicity_V2': ETHNICITY_V2_MODEL,
        'Gender_V3': GENDER_V3_MODEL,
        'Age_V3': AGE_V3_MODEL,
        'Ethnicity_V3': ETHNICITY_V3_MODEL,
    }

    for name, path in model_files.items():
        if os.path.exists(path):
            models[name] = load_interpreter(path)
            print(f"  ✓ {name} chargé")
        else:
            print(f"  ✗ {name} introuvable: {path}")

    # Sample images
    all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    # Filtre: uniquement races 0-3 (pas "Autre")
    valid_images = []
    for f in all_images:
        try:
            age, gender, race = get_label_from_filename(f)
            if 0 <= race <= 3 and 0 <= age <= 116:
                valid_images.append(f)
        except:
            continue

    random.seed(42)
    sample = random.sample(valid_images, min(n_samples, len(valid_images)))

    # Results
    results = {
        'MoE': {'gender_ok': 0, 'eth_ok': 0, 'age_err': [], 'n': 0},
        'Multitask_V4': {'gender_ok': 0, 'eth_ok': 0, 'age_err': [], 'n': 0},
        'Multitask_V5': {'gender_ok': 0, 'eth_ok': 0, 'age_err': [], 'n': 0},
        'Oriented_V2': {'gender_ok': 0, 'eth_ok': 0, 'age_err': [], 'n': 0},
        'Oriented_V3': {'gender_ok': 0, 'eth_ok': 0, 'age_err': [], 'n': 0},
    }

    # MoE per-category tracking
    moe_by_gender = {0: {'ok': 0, 'total': 0}, 1: {'ok': 0, 'total': 0}}
    moe_by_eth = {i: {'ok': 0, 'total': 0} for i in range(4)}
    moe_age_by_bracket = {}

    moe_outputs = models['MoE'].get_output_details()

    for img_file in sample:
        try:
            true_age, true_gender, true_race = get_label_from_filename(img_file)
            img = preprocess_image(os.path.join(DATA_DIR, img_file))
            img_input = np.expand_dims(img, axis=0)

            # === MoE ===
            interp = models['MoE']
            interp.set_tensor(interp.get_input_details()[0]['index'], img_input)
            interp.invoke()

            moe_age = int(np.clip(interp.get_tensor(moe_outputs[0]['index'])[0][0], 0, 116))
            moe_gender_prob = interp.get_tensor(moe_outputs[1]['index'])[0][0]
            moe_eth_probs = interp.get_tensor(moe_outputs[2]['index'])[0]
            moe_gender = 1 if moe_gender_prob > 0.5 else 0
            moe_eth = int(np.argmax(moe_eth_probs))

            results['MoE']['n'] += 1
            if moe_gender == true_gender:
                results['MoE']['gender_ok'] += 1
            if moe_eth == true_race:
                results['MoE']['eth_ok'] += 1
            results['MoE']['age_err'].append(abs(moe_age - true_age))

            # MoE detailed
            moe_by_gender[true_gender]['total'] += 1
            if moe_gender == true_gender:
                moe_by_gender[true_gender]['ok'] += 1
            moe_by_eth[true_race]['total'] += 1
            if moe_eth == true_race:
                moe_by_eth[true_race]['ok'] += 1

            bracket = f"{(true_age // 10) * 10}-{(true_age // 10) * 10 + 9}"
            if bracket not in moe_age_by_bracket:
                moe_age_by_bracket[bracket] = []
            moe_age_by_bracket[bracket].append(abs(moe_age - true_age))

            # === Multitask V4 ===
            if 'Multitask_V4' in models:
                interp = models['Multitask_V4']
                interp.set_tensor(interp.get_input_details()[0]['index'], img_input)
                interp.invoke()
                outs = interp.get_output_details()
                mt_gender_prob = interp.get_tensor(outs[0]['index'])[0][0]
                mt_age = int(np.clip(interp.get_tensor(outs[1]['index'])[0][0], 0, 116))
                mt_eth_probs = interp.get_tensor(outs[2]['index'])[0]
                mt_gender = 1 if mt_gender_prob > 0.5 else 0
                mt_eth = int(np.argmax(mt_eth_probs))

                results['Multitask_V4']['n'] += 1
                if mt_gender == true_gender: results['Multitask_V4']['gender_ok'] += 1
                if mt_eth == true_race: results['Multitask_V4']['eth_ok'] += 1
                results['Multitask_V4']['age_err'].append(abs(mt_age - true_age))

            # === Multitask V5 ===
            if 'Multitask_V5' in models:
                interp = models['Multitask_V5']
                interp.set_tensor(interp.get_input_details()[0]['index'], img_input)
                interp.invoke()
                outs = interp.get_output_details()
                # V5 output order might differ, inspect
                v5_outs = {}
                for o in outs:
                    val = interp.get_tensor(o['index'])
                    if o['shape'][-1] == 4:
                        v5_outs['eth'] = val[0]
                    elif 'gender' in o['name'].lower() or (o['shape'][-1] == 1 and 'age' not in o['name'].lower()):
                        if 'eth' not in v5_outs or o['shape'][-1] == 1:
                            if val[0][0] >= 0 and val[0][0] <= 1.5:
                                v5_outs.setdefault('candidates', []).append(('gender_maybe', val[0][0], o))

                # Fallback: use shape-based heuristic
                vals = []
                for o in outs:
                    v = interp.get_tensor(o['index'])[0]
                    vals.append((o, v))

                v5_gender = None
                v5_age = None
                v5_eth = None
                for o, v in vals:
                    if len(v) == 4:
                        v5_eth = int(np.argmax(v))
                    elif len(v) == 1:
                        if 0 <= v[0] <= 1:
                            v5_gender = 1 if v[0] > 0.5 else 0
                        else:
                            v5_age = int(np.clip(v[0], 0, 116))

                # If both are in [0,1], distinguish by name
                if v5_gender is None or v5_age is None:
                    for o, v in vals:
                        if len(v) == 1 and v5_age is None:
                            v5_age = int(np.clip(v[0], 0, 116))
                        if len(v) == 1 and v5_gender is None:
                            v5_gender = 1 if v[0] > 0.5 else 0

                if v5_gender is not None and v5_age is not None and v5_eth is not None:
                    results['Multitask_V5']['n'] += 1
                    if v5_gender == true_gender: results['Multitask_V5']['gender_ok'] += 1
                    if v5_eth == true_race: results['Multitask_V5']['eth_ok'] += 1
                    results['Multitask_V5']['age_err'].append(abs(v5_age - true_age))

            # === Oriented V2 ===
            if all(k in models for k in ['Gender_V2', 'Age_V2', 'Ethnicity_V2']):
                # Gender V2
                interp_g = models['Gender_V2']
                interp_g.set_tensor(interp_g.get_input_details()[0]['index'], img_input)
                interp_g.invoke()
                g_prob = interp_g.get_tensor(interp_g.get_output_details()[0]['index'])[0][0]
                v2_gender = 1 if g_prob > 0.5 else 0

                # Age V2
                interp_a = models['Age_V2']
                interp_a.set_tensor(interp_a.get_input_details()[0]['index'], img_input)
                interp_a.invoke()
                v2_age = int(np.clip(interp_a.get_tensor(interp_a.get_output_details()[0]['index'])[0][0], 0, 116))

                # Ethnicity V2
                interp_e = models['Ethnicity_V2']
                interp_e.set_tensor(interp_e.get_input_details()[0]['index'], img_input)
                interp_e.invoke()
                eth_probs = interp_e.get_tensor(interp_e.get_output_details()[0]['index'])[0]
                v2_eth = int(np.argmax(eth_probs[:4]))  # Only first 4 classes

                results['Oriented_V2']['n'] += 1
                if v2_gender == true_gender: results['Oriented_V2']['gender_ok'] += 1
                if v2_eth == true_race: results['Oriented_V2']['eth_ok'] += 1
                results['Oriented_V2']['age_err'].append(abs(v2_age - true_age))

            # === Oriented V3 ===
            if all(k in models for k in ['Gender_V3', 'Age_V3', 'Ethnicity_V3']):
                # Gender V3
                interp_g = models['Gender_V3']
                interp_g.set_tensor(interp_g.get_input_details()[0]['index'], img_input)
                interp_g.invoke()
                g_prob = interp_g.get_tensor(interp_g.get_output_details()[0]['index'])[0][0]
                v3_gender = 1 if g_prob > 0.5 else 0

                # Age V3
                interp_a = models['Age_V3']
                interp_a.set_tensor(interp_a.get_input_details()[0]['index'], img_input)
                interp_a.invoke()
                v3_age = int(np.clip(interp_a.get_tensor(interp_a.get_output_details()[0]['index'])[0][0], 0, 116))

                # Ethnicity V3
                interp_e = models['Ethnicity_V3']
                interp_e.set_tensor(interp_e.get_input_details()[0]['index'], img_input)
                interp_e.invoke()
                eth_probs = interp_e.get_tensor(interp_e.get_output_details()[0]['index'])[0]
                v3_eth = int(np.argmax(eth_probs[:4]))

                results['Oriented_V3']['n'] += 1
                if v3_gender == true_gender: results['Oriented_V3']['gender_ok'] += 1
                if v3_eth == true_race: results['Oriented_V3']['eth_ok'] += 1
                results['Oriented_V3']['age_err'].append(abs(v3_age - true_age))

        except Exception as e:
            continue

    # === PRINT RESULTS ===
    print("\n" + "="*60)
    print("RESULTATS COMPARATIFS")
    print("="*60)
    print(f"\n{'Modèle':<25} {'Genre%':>8} {'Ethnie%':>8} {'Age MAE':>8}  {'N':>4}")
    print("-"*60)

    for mode, data in sorted(results.items()):
        if data['n'] == 0:
            continue
        g_acc = data['gender_ok'] / data['n'] * 100
        e_acc = data['eth_ok'] / data['n'] * 100
        age_mae = np.mean(data['age_err'])
        print(f"  {mode:<23} {g_acc:>7.1f}% {e_acc:>7.1f}% {age_mae:>7.2f}  {data['n']:>4}")

    # MoE detailed
    print("\n" + "="*60)
    print("DETAIL MoE PAR CATEGORIE")
    print("="*60)

    print("\n  Genre:")
    for g_id, label in [(0, "Homme"), (1, "Femme")]:
        d = moe_by_gender[g_id]
        if d['total'] > 0:
            print(f"    {label}: {d['ok']}/{d['total']} ({d['ok']/d['total']*100:.1f}%)")

    print("\n  Ethnicité:")
    for e_id, label in enumerate(ETHNICITY_LABELS):
        d = moe_by_eth[e_id]
        if d['total'] > 0:
            print(f"    {label}: {d['ok']}/{d['total']} ({d['ok']/d['total']*100:.1f}%)")

    print("\n  Age MAE par tranche:")
    for bracket in sorted(moe_age_by_bracket.keys()):
        errs = moe_age_by_bracket[bracket]
        print(f"    {bracket}: MAE={np.mean(errs):.1f} (n={len(errs)})")


def test_moe_raw_output_distribution(moe_interp, n=200):
    """Vérifie la distribution des sorties brutes du MoE"""
    print("\n" + "="*60)
    print(f"DISTRIBUTION DES SORTIES BRUTES MoE ({n} images)")
    print("="*60)

    outputs = moe_interp.get_output_details()
    all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    random.seed(123)
    sample = random.sample(all_images, min(n, len(all_images)))

    out0_vals = []
    out1_vals = []
    out2_vals = []

    for img_file in sample:
        try:
            img = preprocess_image(os.path.join(DATA_DIR, img_file))
            img_input = np.expand_dims(img, axis=0)
            moe_interp.set_tensor(moe_interp.get_input_details()[0]['index'], img_input)
            moe_interp.invoke()

            out0_vals.append(moe_interp.get_tensor(outputs[0]['index'])[0].copy())
            out1_vals.append(moe_interp.get_tensor(outputs[1]['index'])[0].copy())
            out2_vals.append(moe_interp.get_tensor(outputs[2]['index'])[0].copy())
        except:
            continue

    out0_vals = np.array(out0_vals)
    out1_vals = np.array(out1_vals)
    out2_vals = np.array(out2_vals)

    print(f"\n  Output[0] ({outputs[0]['name']}):")
    print(f"    Shape par sample: {out0_vals[0].shape}")
    print(f"    Min={out0_vals.min():.4f}, Max={out0_vals.max():.4f}, Mean={out0_vals.mean():.4f}, Std={out0_vals.std():.4f}")

    print(f"\n  Output[1] ({outputs[1]['name']}):")
    print(f"    Shape par sample: {out1_vals[0].shape}")
    print(f"    Min={out1_vals.min():.4f}, Max={out1_vals.max():.4f}, Mean={out1_vals.mean():.4f}, Std={out1_vals.std():.4f}")

    print(f"\n  Output[2] ({outputs[2]['name']}):")
    print(f"    Shape par sample: {out2_vals[0].shape}")
    print(f"    Min={out2_vals.min():.4f}, Max={out2_vals.max():.4f}, Mean={out2_vals.mean():.4f}, Std={out2_vals.std():.4f}")

    # Diagnostic
    print("\n--- DIAGNOSTIC MAPPING ---")

    # Output[0]: si c'est l'âge, devrait être entre 0 et ~100
    # Output[1]: si c'est le genre (sigmoid), devrait être entre 0 et 1
    # Output[2]: si c'est l'ethnicité (softmax), somme devrait être ~1

    if out0_vals.shape[1] == 1:
        v = out0_vals.flatten()
        if v.min() >= -5 and v.max() <= 1.5:
            print(f"  Output[0]: RESSEMBLE A GENDER (sigmoid) [range: {v.min():.2f} - {v.max():.2f}]")
        elif v.min() >= -20 and v.max() <= 150:
            print(f"  Output[0]: RESSEMBLE A AGE (regression) [range: {v.min():.2f} - {v.max():.2f}]")
        else:
            print(f"  Output[0]: RANGE ANORMAL [{v.min():.2f} - {v.max():.2f}]")

    if out1_vals.shape[1] == 1:
        v = out1_vals.flatten()
        if v.min() >= -5 and v.max() <= 1.5:
            print(f"  Output[1]: RESSEMBLE A GENDER (sigmoid) [range: {v.min():.2f} - {v.max():.2f}]")
        elif v.min() >= -20 and v.max() <= 150:
            print(f"  Output[1]: RESSEMBLE A AGE (regression) [range: {v.min():.2f} - {v.max():.2f}]")
        else:
            print(f"  Output[1]: RANGE ANORMAL [{v.min():.2f} - {v.max():.2f}]")

    if out2_vals.shape[1] == 4:
        sums = out2_vals.sum(axis=1)
        print(f"  Output[2]: {out2_vals.shape[1]} classes, somme moyenne={sums.mean():.4f} (softmax attendu: ~1.0)")
        if abs(sums.mean() - 1.0) < 0.1:
            print(f"  Output[2]: RESSEMBLE A ETHNICITY (softmax valide)")
        else:
            print(f"  Output[2]: SOMME ≠ 1, PAS UN SOFTMAX VALIDE!")
    elif out2_vals.shape[1] == 1:
        v = out2_vals.flatten()
        print(f"  Output[2]: Scalaire [range: {v.min():.2f} - {v.max():.2f}]")


def check_rescaling_layer():
    """Vérifie si le modèle MoE contient une couche Rescaling"""
    print("\n" + "="*60)
    print("VÉRIFICATION COUCHE RESCALING DANS MoE")
    print("="*60)

    interp = load_interpreter(MOE_MODEL)

    # Test: envoyer une image noire (tous pixels = 0) vs blanche (tous pixels = 255)
    black_img = np.zeros((1, 128, 128, 3), dtype=np.float32)
    white_img = np.full((1, 128, 128, 3), 255.0, dtype=np.float32)
    gray_img = np.full((1, 128, 128, 3), 127.5, dtype=np.float32)

    outputs = interp.get_output_details()

    for name, img in [("NOIR (0)", black_img), ("GRIS (127.5)", gray_img), ("BLANC (255)", white_img)]:
        interp.set_tensor(interp.get_input_details()[0]['index'], img)
        interp.invoke()

        vals = []
        for o in outputs:
            vals.append(interp.get_tensor(o['index'])[0].copy())

        print(f"\n  Image {name}:")
        for i, v in enumerate(vals):
            print(f"    Output[{i}]: {np.round(v, 4)}")

    # Test: envoyer [0,255] vs [-1,1] et voir lequel donne des résultats sensés
    print("\n--- TEST NORMALISATION ---")
    test_files = [f for f in os.listdir(DATA_DIR) if f.startswith("25_1_0_")]
    if test_files:
        img = preprocess_image(os.path.join(DATA_DIR, test_files[0]))

        # Test 1: [0, 255] (ce que fait l'app)
        img_255 = np.expand_dims(img, axis=0)
        interp.set_tensor(interp.get_input_details()[0]['index'], img_255)
        interp.invoke()
        print(f"\n  Input [0, 255] (comme l'app):")
        for i, o in enumerate(outputs):
            print(f"    Output[{i}]: {interp.get_tensor(o['index'])[0]}")

        # Test 2: [-1, 1] (si Rescaling n'est pas dans le modèle)
        img_norm = (img / 127.5) - 1.0
        img_norm = np.expand_dims(img_norm, axis=0)
        interp.set_tensor(interp.get_input_details()[0]['index'], img_norm)
        interp.invoke()
        print(f"\n  Input [-1, 1] (normalisation manuelle):")
        for i, o in enumerate(outputs):
            print(f"    Output[{i}]: {interp.get_tensor(o['index'])[0]}")

        # Test 3: [0, 1]
        img_01 = img / 255.0
        img_01 = np.expand_dims(img_01, axis=0)
        interp.set_tensor(interp.get_input_details()[0]['index'], img_01)
        interp.invoke()
        print(f"\n  Input [0, 1] (autre normalisation possible):")
        for i, o in enumerate(outputs):
            print(f"    Output[{i}]: {interp.get_tensor(o['index'])[0]}")


def main():
    print("="*60)
    print("DIAGNOSTIC COMPLET DES MODELES TFLITE")
    print("="*60)

    # 1. Inspect all models
    print("\n\n" + "#"*60)
    print("# PARTIE 1: INSPECTION DES MODELES")
    print("#"*60)

    moe_interp = inspect_model("MoE MobileNetV3", MOE_MODEL)
    inspect_model("Multitask V4 (EfficientNet)", MULTITASK_MODEL)
    inspect_model("Multitask V5 (MobileNetV3)", MULTITASK_V5_MODEL)
    inspect_model("Gender V3 (MobileNetV3)", GENDER_V3_MODEL)
    inspect_model("Age V3 (MobileNetV3)", AGE_V3_MODEL)
    inspect_model("Ethnicity V3 (MobileNetV3)", ETHNICITY_V3_MODEL)

    # 2. Check rescaling
    print("\n\n" + "#"*60)
    print("# PARTIE 2: VERIFICATION RESCALING")
    print("#"*60)
    check_rescaling_layer()

    # 3. Test output order
    print("\n\n" + "#"*60)
    print("# PARTIE 3: VERIFICATION ORDRE DES SORTIES MoE")
    print("#"*60)
    test_moe_output_order(moe_interp)

    # 4. Raw output distribution
    print("\n\n" + "#"*60)
    print("# PARTIE 4: DISTRIBUTION DES SORTIES MoE")
    print("#"*60)
    test_moe_raw_output_distribution(moe_interp, n=300)

    # 5. Known images
    print("\n\n" + "#"*60)
    print("# PARTIE 5: TEST SUR IMAGES CONNUES")
    print("#"*60)
    test_moe_with_known_images(moe_interp)

    # 6. Full comparison
    print("\n\n" + "#"*60)
    print("# PARTIE 6: COMPARAISON COMPLETE")
    print("#"*60)
    test_all_models_comparison(n_samples=500)

    print("\n\nDIAGNOSTIC TERMINE.")

if __name__ == "__main__":
    main()
