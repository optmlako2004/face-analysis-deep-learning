#!/usr/bin/env python3
"""
Simulation EXACTE du pipeline Android pour identifier la source des mauvaises prédictions MoE.

Reproduit pas à pas ce que fait l'app:
  1. Image brute (comme la caméra)
  2. Détection de visage + crop avec padding 0.5
  3. Resize 128x128
  4. bitmapToByteBuffer: RGB float32 [0,255]
  5. TTA: original + flip horizontal
  6. Moyenne des 2 inférences
  7. EMA smoothing (alpha=0.4) sur séquence
"""

import numpy as np
import tensorflow as tf
from PIL import Image, ImageFilter
import os
import random

# ============================================================
# CONFIG
# ============================================================
MODEL_DIR = "FacePredictor/app/src/main/assets"
DATA_DIR = "data/UTKFace"
MOE_MODEL = os.path.join(MODEL_DIR, "moe_mobilenetv3.tflite")

ETHNICITY_LABELS = ["Blanc", "Noir", "Asiatique", "Indien"]

def load_interpreter(path):
    interp = tf.lite.Interpreter(model_path=path)
    interp.allocate_tensors()
    return interp


# ============================================================
# SIMULATION DES ETAPES DU PIPELINE ANDROID
# ============================================================

def simulate_camera_degradation(img_array):
    """
    Simule ce que la caméra fait à l'image:
    - Compression JPEG (comme imageProxy.toBitmap)
    - Légère perte de qualité
    """
    from io import BytesIO
    img = Image.fromarray(img_array.astype(np.uint8))
    buffer = BytesIO()
    img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    img_compressed = Image.open(buffer)
    return np.array(img_compressed, dtype=np.float32)


def simulate_face_crop_with_padding(img_array, padding=0.5):
    """
    Simule le crop de FaceDetectorHelper.cropFace() avec padding.
    Les images UTKFace sont déjà des visages croppés, donc on simule
    ce qui se passe quand le crop MediaPipe est légèrement différent:
    - Crop plus serré (comme si MediaPipe détectait un bbox plus petit)
    - Crop décentré (comme si le visage n'était pas centré)
    """
    h, w = img_array.shape[:2]
    # Simuler un crop MediaPipe qui détecte ~70% du visage
    # puis ajoute le padding de 0.5
    face_w = int(w * 0.7)
    face_h = int(h * 0.7)
    face_x = (w - face_w) // 2
    face_y = (h - face_h) // 2

    # Ajouter padding (comme l'app)
    pad_x = int(face_w * padding)
    pad_y = int(face_h * padding)
    left = max(0, face_x - pad_x)
    top = max(0, face_y - pad_y)
    right = min(w, face_x + face_w + pad_x)
    bottom = min(h, face_y + face_h + pad_y)

    return img_array[top:bottom, left:right]


def simulate_face_crop_square(img_array, padding=0.15):
    """
    Simule le NOUVEAU cropFaceSquare() de FaceDetector.kt.
    - Prend max(w,h) du bbox comme base
    - Applique padding symétriquement pour obtenir un carré
    - Décale avant de clipper pour garder le carré
    """
    h, w = img_array.shape[:2]
    # Simuler un bbox MediaPipe qui détecte ~70% du visage
    # (souvent plus haut que large)
    face_w = int(w * 0.7)
    face_h = int(h * 0.75)  # bbox souvent plus haute que large
    face_x = (w - face_w) // 2
    face_y = (h - face_h) // 2

    # cropFaceSquare: max(w,h) comme base
    base_side = max(face_w, face_h)
    crop_side = int(base_side * (1 + 2 * padding))

    # Centre du bbox
    cx = face_x + face_w // 2
    cy = face_y + face_h // 2

    # Top-left corner
    left = cx - crop_side // 2
    top = cy - crop_side // 2

    # Shift to stay within bounds
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if left + crop_side > w:
        left = max(0, w - crop_side)
    if top + crop_side > h:
        top = max(0, h - crop_side)

    # Final square
    final_w = min(crop_side, w - left)
    final_h = min(crop_side, h - top)
    final_side = min(final_w, final_h)

    if final_side > 0:
        return img_array[top:top+final_side, left:left+final_side]
    return img_array


def simulate_different_crop_ratios(img_array):
    """
    Simule différents ratios de crop pour voir l'impact.
    Retourne un dict de crops avec différentes quantités de contexte.
    """
    h, w = img_array.shape[:2]
    crops = {}

    # 1. Image entière (comme UTKFace brut)
    crops["100% (brut UTKFace)"] = img_array

    # 2. Crop 90% centré
    margin = 0.05
    l, t = int(w*margin), int(h*margin)
    r, b = int(w*(1-margin)), int(h*(1-margin))
    crops["90% centre"] = img_array[t:b, l:r]

    # 3. Crop 70% centré (visage serré)
    margin = 0.15
    l, t = int(w*margin), int(h*margin)
    r, b = int(w*(1-margin)), int(h*(1-margin))
    crops["70% centre"] = img_array[t:b, l:r]

    # 4. Crop 50% centré (très serré)
    margin = 0.25
    l, t = int(w*margin), int(h*margin)
    r, b = int(w*(1-margin)), int(h*(1-margin))
    crops["50% centre"] = img_array[t:b, l:r]

    # 5. Crop décentré (décalé vers la droite)
    margin = 0.1
    l = int(w * 0.2)
    t = int(h * margin)
    r = min(w, l + int(w * 0.7))
    b = int(h * (1-margin))
    crops["70% decentre droite"] = img_array[t:b, l:r]

    # 6. Crop avec beaucoup de contexte (padding large)
    # Simuler en ajoutant du noir autour
    padded = np.zeros((int(h*1.5), int(w*1.5), 3), dtype=np.float32)
    offset_y, offset_x = int(h*0.25), int(w*0.25)
    padded[offset_y:offset_y+h, offset_x:offset_x+w] = img_array
    crops["Avec contexte noir"] = padded

    return crops


def predict_moe_single(interp, img_array):
    """Prédiction MoE simple (sans TTA), identique à l'app sans flip."""
    img_resized = np.array(Image.fromarray(img_array.astype(np.uint8)).resize((128, 128)), dtype=np.float32)
    img_input = np.expand_dims(img_resized, axis=0)

    outputs = interp.get_output_details()
    interp.set_tensor(interp.get_input_details()[0]['index'], img_input)
    interp.invoke()

    age = float(interp.get_tensor(outputs[0]['index'])[0][0])
    gender_prob = float(interp.get_tensor(outputs[1]['index'])[0][0])
    eth_probs = interp.get_tensor(outputs[2]['index'])[0].copy()

    return age, gender_prob, eth_probs


def predict_moe_tta(interp, img_array):
    """Prédiction MoE avec TTA (original + flip), comme le code Kotlin."""
    img_pil = Image.fromarray(img_array.astype(np.uint8)).resize((128, 128))
    img_resized = np.array(img_pil, dtype=np.float32)
    img_flipped = np.flip(img_resized, axis=1).copy()

    outputs = interp.get_output_details()

    # Original
    interp.set_tensor(interp.get_input_details()[0]['index'], np.expand_dims(img_resized, 0))
    interp.invoke()
    age1 = float(interp.get_tensor(outputs[0]['index'])[0][0])
    gender1 = float(interp.get_tensor(outputs[1]['index'])[0][0])
    eth1 = interp.get_tensor(outputs[2]['index'])[0].copy()

    # Flipped
    interp.set_tensor(interp.get_input_details()[0]['index'], np.expand_dims(img_flipped, 0))
    interp.invoke()
    age2 = float(interp.get_tensor(outputs[0]['index'])[0][0])
    gender2 = float(interp.get_tensor(outputs[1]['index'])[0][0])
    eth2 = interp.get_tensor(outputs[2]['index'])[0].copy()

    # Moyenne TTA
    age_avg = (age1 + age2) / 2
    gender_avg = (gender1 + gender2) / 2
    eth_avg = (eth1 + eth2) / 2

    return age_avg, gender_avg, eth_avg


def apply_ema_smoothing(results_sequence, alpha=0.4):
    """
    Applique EMA sur une séquence de résultats, comme RealtimeCameraActivity.
    """
    ema_age = None
    ema_gender = None
    ema_eth = None
    smoothed = []

    for age, gender_prob, eth_probs in results_sequence:
        if ema_age is None:
            ema_age = age
            ema_gender = gender_prob
            ema_eth = eth_probs.copy()
        else:
            ema_age = alpha * age + (1 - alpha) * ema_age
            ema_gender = alpha * gender_prob + (1 - alpha) * ema_gender
            ema_eth = alpha * eth_probs + (1 - alpha) * ema_eth

        smoothed.append((ema_age, ema_gender, ema_eth.copy()))

    return smoothed


# ============================================================
# TESTS
# ============================================================

def test_1_crop_impact(interp, n=100):
    """Test 1: Impact du cropping sur les prédictions"""
    print("\n" + "="*70)
    print("TEST 1: IMPACT DU CROP SUR LES PREDICTIONS MoE")
    print("="*70)
    print("(Compare image brute UTKFace vs différents niveaux de crop)")

    all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    valid = []
    for f in all_images:
        try:
            parts = f.split('_')
            age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= race <= 3:
                valid.append((f, age, gender, race))
        except:
            continue

    random.seed(42)
    sample = random.sample(valid, min(n, len(valid)))

    crop_results = {}

    for img_file, true_age, true_gender, true_race in sample:
        img = np.array(Image.open(os.path.join(DATA_DIR, img_file)).convert('RGB'), dtype=np.float32)
        crops = simulate_different_crop_ratios(img)

        for crop_name, cropped in crops.items():
            if crop_name not in crop_results:
                crop_results[crop_name] = {'gender_ok': 0, 'eth_ok': 0, 'age_err': [], 'n': 0}

            try:
                age, gender_prob, eth_probs = predict_moe_single(interp, cropped)
                pred_gender = 1 if gender_prob > 0.5 else 0
                pred_eth = int(np.argmax(eth_probs))
                pred_age = int(np.clip(age, 0, 116))

                crop_results[crop_name]['n'] += 1
                if pred_gender == true_gender:
                    crop_results[crop_name]['gender_ok'] += 1
                if pred_eth == true_race:
                    crop_results[crop_name]['eth_ok'] += 1
                crop_results[crop_name]['age_err'].append(abs(pred_age - true_age))
            except:
                continue

    print(f"\n{'Crop':<30} {'Genre%':>8} {'Ethnie%':>8} {'Age MAE':>8}")
    print("-"*60)
    for crop_name, data in crop_results.items():
        if data['n'] == 0:
            continue
        g = data['gender_ok'] / data['n'] * 100
        e = data['eth_ok'] / data['n'] * 100
        a = np.mean(data['age_err'])
        print(f"  {crop_name:<28} {g:>7.1f}% {e:>7.1f}% {a:>7.2f}")


def test_2_jpeg_compression_impact(interp, n=100):
    """Test 2: Impact de la compression JPEG (comme la caméra)"""
    print("\n" + "="*70)
    print("TEST 2: IMPACT DE LA COMPRESSION JPEG")
    print("="*70)

    all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    valid = []
    for f in all_images:
        try:
            parts = f.split('_')
            age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= race <= 3:
                valid.append((f, age, gender, race))
        except:
            continue

    random.seed(42)
    sample = random.sample(valid, min(n, len(valid)))

    qualities = {'Original (pas de compression)': None, 'JPEG Q=95': 95, 'JPEG Q=85': 85,
                 'JPEG Q=70': 70, 'JPEG Q=50': 50}

    results = {q: {'gender_ok': 0, 'eth_ok': 0, 'age_err': [], 'n': 0} for q in qualities}

    for img_file, true_age, true_gender, true_race in sample:
        img = np.array(Image.open(os.path.join(DATA_DIR, img_file)).convert('RGB'), dtype=np.float32)

        for q_name, quality in qualities.items():
            try:
                if quality is None:
                    test_img = img
                else:
                    test_img = simulate_camera_jpeg(img, quality)

                age, gender_prob, eth_probs = predict_moe_single(interp, test_img)
                pred_gender = 1 if gender_prob > 0.5 else 0
                pred_eth = int(np.argmax(eth_probs))
                pred_age = int(np.clip(age, 0, 116))

                results[q_name]['n'] += 1
                if pred_gender == true_gender:
                    results[q_name]['gender_ok'] += 1
                if pred_eth == true_race:
                    results[q_name]['eth_ok'] += 1
                results[q_name]['age_err'].append(abs(pred_age - true_age))
            except:
                continue

    print(f"\n{'Qualité':<35} {'Genre%':>8} {'Ethnie%':>8} {'Age MAE':>8}")
    print("-"*65)
    for q_name, data in results.items():
        if data['n'] == 0:
            continue
        g = data['gender_ok'] / data['n'] * 100
        e = data['eth_ok'] / data['n'] * 100
        a = np.mean(data['age_err'])
        print(f"  {q_name:<33} {g:>7.1f}% {e:>7.1f}% {a:>7.2f}")


def simulate_camera_jpeg(img_array, quality=85):
    from io import BytesIO
    img = Image.fromarray(img_array.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return np.array(Image.open(buf).convert('RGB'), dtype=np.float32)


def test_3_tta_impact(interp, n=200):
    """Test 3: Impact du TTA (flip)"""
    print("\n" + "="*70)
    print("TEST 3: IMPACT DU TTA (original + flip horizontal)")
    print("="*70)

    all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    valid = []
    for f in all_images:
        try:
            parts = f.split('_')
            age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= race <= 3:
                valid.append((f, age, gender, race))
        except:
            continue

    random.seed(42)
    sample = random.sample(valid, min(n, len(valid)))

    res_single = {'gender_ok': 0, 'eth_ok': 0, 'age_err': []}
    res_tta = {'gender_ok': 0, 'eth_ok': 0, 'age_err': []}
    count = 0

    for img_file, true_age, true_gender, true_race in sample:
        try:
            img = np.array(Image.open(os.path.join(DATA_DIR, img_file)).convert('RGB'), dtype=np.float32)

            # Sans TTA
            age1, gp1, ep1 = predict_moe_single(interp, img)
            pg1 = 1 if gp1 > 0.5 else 0
            pe1 = int(np.argmax(ep1))
            pa1 = int(np.clip(age1, 0, 116))

            # Avec TTA
            age2, gp2, ep2 = predict_moe_tta(interp, img)
            pg2 = 1 if gp2 > 0.5 else 0
            pe2 = int(np.argmax(ep2))
            pa2 = int(np.clip(age2, 0, 116))

            count += 1
            if pg1 == true_gender: res_single['gender_ok'] += 1
            if pe1 == true_race: res_single['eth_ok'] += 1
            res_single['age_err'].append(abs(pa1 - true_age))

            if pg2 == true_gender: res_tta['gender_ok'] += 1
            if pe2 == true_race: res_tta['eth_ok'] += 1
            res_tta['age_err'].append(abs(pa2 - true_age))
        except:
            continue

    print(f"\n{'Mode':<25} {'Genre%':>8} {'Ethnie%':>8} {'Age MAE':>8}")
    print("-"*55)
    for name, data in [("Sans TTA", res_single), ("Avec TTA (flip)", res_tta)]:
        g = data['gender_ok'] / count * 100
        e = data['eth_ok'] / count * 100
        a = np.mean(data['age_err'])
        print(f"  {name:<23} {g:>7.1f}% {e:>7.1f}% {a:>7.2f}")


def test_4_ema_smoothing_simulation(interp):
    """Test 4: Simulation EMA smoothing sur une séquence (comme en temps réel)"""
    print("\n" + "="*70)
    print("TEST 4: SIMULATION EMA SMOOTHING (séquence temps réel)")
    print("="*70)
    print("Simule ce qui se passe quand la caméra filme la même personne")

    # Prendre 5 personnes différentes, simuler une séquence de 10 frames
    test_cases = [
        ("25_1_0_", "25 ans, Femme, Blanche"),
        ("40_0_1_", "40 ans, Homme, Noir"),
        ("10_0_2_", "10 ans, Homme, Asiatique"),
        ("30_1_3_", "30 ans, Femme, Indienne"),
        ("55_0_0_", "55 ans, Homme, Blanc"),
    ]

    for prefix, desc in test_cases:
        files = [f for f in os.listdir(DATA_DIR) if f.startswith(prefix)]
        if not files:
            continue

        img = np.array(Image.open(os.path.join(DATA_DIR, files[0])).convert('RGB'), dtype=np.float32)

        # Simuler 10 frames avec légères variations (crop, bruit)
        frames = []
        for i in range(10):
            # Ajouter petit bruit aléatoire (comme les variations de frame en frame)
            noise = np.random.normal(0, 3, img.shape).astype(np.float32)
            noisy = np.clip(img + noise, 0, 255)

            # Léger décalage de crop aléatoire
            h, w = noisy.shape[:2]
            dx, dy = random.randint(-3, 3), random.randint(-3, 3)
            margin = 5
            l = max(0, margin + dx)
            t = max(0, margin + dy)
            r = min(w, w - margin + dx)
            b = min(h, h - margin + dy)
            cropped = noisy[t:b, l:r]

            age, gp, ep = predict_moe_tta(interp, cropped)
            frames.append((age, gp, ep))

        # Appliquer EMA
        smoothed = apply_ema_smoothing(frames, alpha=0.4)

        print(f"\n  Personne: {desc} (fichier: {files[0]})")
        print(f"  {'Frame':<8} {'Age':>5} {'Gender prob':>12} {'Pred Genre':>12} {'Top Eth':>12} {'Eth conf':>9}")
        print(f"  " + "-"*60)

        parts = files[0].split('_')
        true_age, true_gender, true_race = int(parts[0]), int(parts[1]), int(parts[2])

        for i, ((raw_age, raw_gp, raw_ep), (sm_age, sm_gp, sm_ep)) in enumerate(zip(frames, smoothed)):
            pred_gender = "Femme" if sm_gp > 0.5 else "Homme"
            pred_eth = ETHNICITY_LABELS[int(np.argmax(sm_ep))]
            eth_conf = sm_ep[int(np.argmax(sm_ep))]
            marker = ""
            if (1 if sm_gp > 0.5 else 0) != true_gender:
                marker += " !! GENRE"
            if int(np.argmax(sm_ep)) != true_race:
                marker += " !! ETH"
            print(f"  {i+1:<8} {int(sm_age):>5} {sm_gp:>12.4f} {pred_gender:>12} {pred_eth:>12} {eth_conf:>8.3f}{marker}")


def test_5_camera_simulation_full_pipeline(interp, n=200):
    """
    Test 5: Pipeline COMPLET simulant l'app Android
    Compare 3 modes:
      A. Image brute + TTA (référence Python)
      B. ANCIEN pipeline: JPEG + cropFace(padding=0.5) non carré + TTA
      C. NOUVEAU pipeline: JPEG + cropFaceSquare(padding=0.15) carré + TTA
    """
    print("\n" + "="*70)
    print("TEST 5: PIPELINE COMPLET — ANCIEN vs NOUVEAU CROP")
    print("="*70)

    all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    valid = []
    for f in all_images:
        try:
            parts = f.split('_')
            age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= race <= 3:
                valid.append((f, age, gender, race))
        except:
            continue

    random.seed(42)
    sample = random.sample(valid, min(n, len(valid)))

    res_raw = {'gender_ok': 0, 'eth_ok': 0, 'age_err': []}
    res_old_pipeline = {'gender_ok': 0, 'eth_ok': 0, 'age_err': []}
    res_new_pipeline = {'gender_ok': 0, 'eth_ok': 0, 'age_err': []}
    count = 0

    disagree_examples = []

    for img_file, true_age, true_gender, true_race in sample:
        try:
            img = np.array(Image.open(os.path.join(DATA_DIR, img_file)).convert('RGB'), dtype=np.float32)

            # A. Test RAW (image directe, référence Python)
            age_r, gp_r, ep_r = predict_moe_tta(interp, img)
            pg_r = 1 if gp_r > 0.5 else 0
            pe_r = int(np.argmax(ep_r))
            pa_r = int(np.clip(age_r, 0, 116))

            # B. ANCIEN pipeline (cropFace padding=0.5, non carré)
            img_jpeg = simulate_camera_jpeg(img, 85)
            img_old = simulate_face_crop_with_padding(img_jpeg, padding=0.5)
            age_o, gp_o, ep_o = predict_moe_tta(interp, img_old)
            pg_o = 1 if gp_o > 0.5 else 0
            pe_o = int(np.argmax(ep_o))
            pa_o = int(np.clip(age_o, 0, 116))

            # C. NOUVEAU pipeline (cropFaceSquare padding=0.15, carré)
            img_new = simulate_face_crop_square(img_jpeg, padding=0.15)
            age_n, gp_n, ep_n = predict_moe_tta(interp, img_new)
            pg_n = 1 if gp_n > 0.5 else 0
            pe_n = int(np.argmax(ep_n))
            pa_n = int(np.clip(age_n, 0, 116))

            count += 1

            if pg_r == true_gender: res_raw['gender_ok'] += 1
            if pe_r == true_race: res_raw['eth_ok'] += 1
            res_raw['age_err'].append(abs(pa_r - true_age))

            if pg_o == true_gender: res_old_pipeline['gender_ok'] += 1
            if pe_o == true_race: res_old_pipeline['eth_ok'] += 1
            res_old_pipeline['age_err'].append(abs(pa_o - true_age))

            if pg_n == true_gender: res_new_pipeline['gender_ok'] += 1
            if pe_n == true_race: res_new_pipeline['eth_ok'] += 1
            res_new_pipeline['age_err'].append(abs(pa_n - true_age))

            # Tracker les améliorations nouveau vs ancien
            old_wrong = (pg_o != true_gender) or (pe_o != true_race)
            new_right = (pg_n == true_gender) and (pe_n == true_race)
            if old_wrong and new_right:
                disagree_examples.append({
                    'file': img_file,
                    'true': f"Age={true_age}, G={'F' if true_gender==1 else 'H'}, E={ETHNICITY_LABELS[true_race]}",
                    'old': f"Age={pa_o}, G={'F' if pg_o==1 else 'H'}, E={ETHNICITY_LABELS[pe_o]}",
                    'new': f"Age={pa_n}, G={'F' if pg_n==1 else 'H'}, E={ETHNICITY_LABELS[pe_n]}",
                })
        except:
            continue

    print(f"\n{'Mode':<45} {'Genre%':>8} {'Ethnie%':>8} {'Age MAE':>8}")
    print("-"*75)
    for name, data in [
        ("A. Image brute + TTA (ref Python)", res_raw),
        ("B. ANCIEN: JPEG+cropFace(0.5)+TTA", res_old_pipeline),
        ("C. NOUVEAU: JPEG+cropFaceSquare(0.15)+TTA", res_new_pipeline),
    ]:
        g = data['gender_ok'] / count * 100
        e = data['eth_ok'] / count * 100
        a = np.mean(data['age_err'])
        print(f"  {name:<43} {g:>7.1f}% {e:>7.1f}% {a:>7.2f}")

    # Deltas
    print("\n  --- Deltas ---")
    old_g = res_old_pipeline['gender_ok'] / count * 100
    new_g = res_new_pipeline['gender_ok'] / count * 100
    old_e = res_old_pipeline['eth_ok'] / count * 100
    new_e = res_new_pipeline['eth_ok'] / count * 100
    old_a = np.mean(res_old_pipeline['age_err'])
    new_a = np.mean(res_new_pipeline['age_err'])
    print(f"  Nouveau vs Ancien:  Genre {new_g - old_g:+.1f}%  Ethnie {new_e - old_e:+.1f}%  Age MAE {new_a - old_a:+.2f}")

    raw_g = res_raw['gender_ok'] / count * 100
    raw_e = res_raw['eth_ok'] / count * 100
    raw_a = np.mean(res_raw['age_err'])
    print(f"  Nouveau vs Ref:     Genre {new_g - raw_g:+.1f}%  Ethnie {new_e - raw_e:+.1f}%  Age MAE {new_a - raw_a:+.2f}")

    if disagree_examples:
        print(f"\n  Exemples corriges par le nouveau crop ({len(disagree_examples)} cas):")
        for ex in disagree_examples[:8]:
            print(f"    {ex['file']}")
            print(f"      Verite: {ex['true']}")
            print(f"      Ancien: {ex['old']}")
            print(f"      Nouveau: {ex['new']}")


def test_6_lighting_conditions(interp, n=100):
    """Test 6: Impact des conditions d'éclairage (problème fréquent en caméra réelle)"""
    print("\n" + "="*70)
    print("TEST 6: IMPACT DES CONDITIONS D'ECLAIRAGE")
    print("="*70)

    all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    valid = []
    for f in all_images:
        try:
            parts = f.split('_')
            age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= race <= 3:
                valid.append((f, age, gender, race))
        except:
            continue

    random.seed(42)
    sample = random.sample(valid, min(n, len(valid)))

    conditions = {
        'Normal': lambda x: x,
        'Sombre (-50)': lambda x: np.clip(x - 50, 0, 255),
        'Sombre (-100)': lambda x: np.clip(x - 100, 0, 255),
        'Lumineux (+50)': lambda x: np.clip(x + 50, 0, 255),
        'Lumineux (+100)': lambda x: np.clip(x + 100, 0, 255),
        'Contraste faible': lambda x: np.clip(x * 0.5 + 64, 0, 255),
        'Contraste fort': lambda x: np.clip((x - 128) * 1.5 + 128, 0, 255),
        'Teinte jaune (lampe)': lambda x: np.clip(np.stack([x[:,:,0]+20, x[:,:,1]+10, x[:,:,2]-20], axis=-1), 0, 255),
    }

    results = {c: {'gender_ok': 0, 'eth_ok': 0, 'age_err': [], 'n': 0} for c in conditions}

    for img_file, true_age, true_gender, true_race in sample:
        img = np.array(Image.open(os.path.join(DATA_DIR, img_file)).convert('RGB'), dtype=np.float32)

        for cond_name, transform in conditions.items():
            try:
                modified = transform(img).astype(np.float32)
                age, gp, ep = predict_moe_single(interp, modified)
                pg = 1 if gp > 0.5 else 0
                pe = int(np.argmax(ep))
                pa = int(np.clip(age, 0, 116))

                results[cond_name]['n'] += 1
                if pg == true_gender: results[cond_name]['gender_ok'] += 1
                if pe == true_race: results[cond_name]['eth_ok'] += 1
                results[cond_name]['age_err'].append(abs(pa - true_age))
            except:
                continue

    print(f"\n{'Condition':<30} {'Genre%':>8} {'Ethnie%':>8} {'Age MAE':>8}")
    print("-"*60)
    for cond_name, data in results.items():
        if data['n'] == 0:
            continue
        g = data['gender_ok'] / data['n'] * 100
        e = data['eth_ok'] / data['n'] * 100
        a = np.mean(data['age_err'])
        marker = ""
        if g < 85 or e < 80:
            marker = "  <-- DEGRADATION"
        print(f"  {cond_name:<28} {g:>7.1f}% {e:>7.1f}% {a:>7.2f}{marker}")


def test_7_front_camera_mirror(interp, n=100):
    """Test 7: Impact du mirroring caméra frontale"""
    print("\n" + "="*70)
    print("TEST 7: IMPACT DU MIRRORING CAMERA FRONTALE")
    print("="*70)

    all_images = [f for f in os.listdir(DATA_DIR) if f.endswith('.jpg')]
    valid = []
    for f in all_images:
        try:
            parts = f.split('_')
            age, gender, race = int(parts[0]), int(parts[1]), int(parts[2])
            if 0 <= race <= 3:
                valid.append((f, age, gender, race))
        except:
            continue

    random.seed(42)
    sample = random.sample(valid, min(n, len(valid)))

    res_normal = {'gender_ok': 0, 'eth_ok': 0, 'age_err': []}
    res_mirrored = {'gender_ok': 0, 'eth_ok': 0, 'age_err': []}
    res_rotated_90 = {'gender_ok': 0, 'eth_ok': 0, 'age_err': []}
    res_rotated_mirror = {'gender_ok': 0, 'eth_ok': 0, 'age_err': []}
    count = 0

    for img_file, true_age, true_gender, true_race in sample:
        try:
            img = np.array(Image.open(os.path.join(DATA_DIR, img_file)).convert('RGB'), dtype=np.float32)

            # Normal
            a, gp, ep = predict_moe_single(interp, img)
            pg = 1 if gp > 0.5 else 0
            pe = int(np.argmax(ep))
            pa = int(np.clip(a, 0, 116))
            count += 1
            if pg == true_gender: res_normal['gender_ok'] += 1
            if pe == true_race: res_normal['eth_ok'] += 1
            res_normal['age_err'].append(abs(pa - true_age))

            # Mirrored horizontally (comme caméra frontale)
            img_mirror = np.flip(img, axis=1).copy()
            a, gp, ep = predict_moe_single(interp, img_mirror)
            pg = 1 if gp > 0.5 else 0
            pe = int(np.argmax(ep))
            pa = int(np.clip(a, 0, 116))
            if pg == true_gender: res_mirrored['gender_ok'] += 1
            if pe == true_race: res_mirrored['eth_ok'] += 1
            res_mirrored['age_err'].append(abs(pa - true_age))

            # Rotated 90 degrees (mauvaise rotation)
            img_rot = np.rot90(img).copy()
            a, gp, ep = predict_moe_single(interp, img_rot)
            pg = 1 if gp > 0.5 else 0
            pe = int(np.argmax(ep))
            pa = int(np.clip(a, 0, 116))
            if pg == true_gender: res_rotated_90['gender_ok'] += 1
            if pe == true_race: res_rotated_90['eth_ok'] += 1
            res_rotated_90['age_err'].append(abs(pa - true_age))

            # Rotated 90 + mirrored (erreur de rotation caméra)
            img_rot_mir = np.flip(np.rot90(img), axis=1).copy()
            a, gp, ep = predict_moe_single(interp, img_rot_mir)
            pg = 1 if gp > 0.5 else 0
            pe = int(np.argmax(ep))
            pa = int(np.clip(a, 0, 116))
            if pg == true_gender: res_rotated_mirror['gender_ok'] += 1
            if pe == true_race: res_rotated_mirror['eth_ok'] += 1
            res_rotated_mirror['age_err'].append(abs(pa - true_age))

        except:
            continue

    print(f"\n{'Transformation':<30} {'Genre%':>8} {'Ethnie%':>8} {'Age MAE':>8}")
    print("-"*60)
    for name, data in [("Normal", res_normal), ("Mirror horizontal", res_mirrored),
                        ("Rotation 90°", res_rotated_90), ("Rotation 90° + mirror", res_rotated_mirror)]:
        g = data['gender_ok'] / count * 100
        e = data['eth_ok'] / count * 100
        a = np.mean(data['age_err'])
        marker = ""
        if g < 80 or e < 70:
            marker = "  <-- PROBLEME!"
        print(f"  {name:<28} {g:>7.1f}% {e:>7.1f}% {a:>7.2f}{marker}")


def main():
    print("="*70)
    print("SIMULATION DU PIPELINE ANDROID - DIAGNOSTIC MoE")
    print("="*70)

    interp = load_interpreter(MOE_MODEL)
    print("MoE model charge\n")

    test_1_crop_impact(interp, n=200)
    test_2_jpeg_compression_impact(interp, n=200)
    test_3_tta_impact(interp, n=300)
    test_4_ema_smoothing_simulation(interp)
    test_5_camera_simulation_full_pipeline(interp, n=300)
    test_6_lighting_conditions(interp, n=200)
    test_7_front_camera_mirror(interp, n=200)

    print("\n" + "="*70)
    print("SIMULATION TERMINEE")
    print("="*70)


if __name__ == "__main__":
    main()
