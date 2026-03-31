#!/usr/bin/env python3
"""
Test Hybrid Mode - Combine les meilleurs modèles pour chaque tâche:
- Genre: V2 Orienté (gender_v2_model.tflite) - 100% correct en tests
- Ethnicité: V4 Multitâche (multitask_model.tflite) - Meilleur pour ethnicité
- Age: V2 Orienté (age_v2_model.tflite)
"""

import numpy as np
import tensorflow as tf
from PIL import Image
import os
import glob
import sys
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
MULTITASK_MODEL = str(MODEL_DIR / "multitask_model.tflite")

# Input sizes will be detected at runtime from the models
INPUT_SIZE = 128

# Labels
GENDER_LABELS = ["Homme", "Femme"]
ETHNICITY_LABELS_V4 = ["Blanc", "Noir", "Asiatique", "Indien"]

def load_interpreter(model_path):
    """Load TFLite interpreter"""
    # Use builtin_ref resolver to support newer TFLite ops with older TF wheels
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
                raise RuntimeError(
                    "FULLY_CONNECTED v12 non pris en charge: installez TensorFlow >= 2.17/ tf-nightly ou assurez-vous que libtensorflowlite_flex.so est disponible"
                ) from exc
        raise

def preprocess_image(image_path, size):
    """Preprocess image for model input (returns float32 HxWx3)."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((size, size))
    img_array = np.array(img, dtype=np.float32)
    return img_array


def resize_img(img_array, size):
    """Resize an HxWx3 array to size x size (float32)."""
    if img_array.shape[0] == size and img_array.shape[1] == size:
        return img_array
    return tf.image.resize(img_array, (size, size)).numpy().astype(np.float32)

def apply_clahe_simple(img_array):
    """Apply simple contrast enhancement (similar to Android CLAHE)"""
    # Compute luminance
    luminance = 0.299 * img_array[:,:,0] + 0.587 * img_array[:,:,1] + 0.114 * img_array[:,:,2]
    min_l = luminance.min()
    max_l = luminance.max()

    if max_l - min_l > 0:
        scale = 255.0 / (max_l - min_l)
        img_enhanced = (img_array - min_l) * scale
        img_enhanced = np.clip(img_enhanced, 0, 255)
        return img_enhanced
    return img_array

def predict_gender_v2(interpreter, img_array):
    """Predict gender using V2 model (with CLAHE)"""
    # Apply CLAHE-like preprocessing
    img_clahe = apply_clahe_simple(img_array)
    img_input = np.expand_dims(img_clahe, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    prob = output[0][0]

    gender = "Femme" if prob > 0.5 else "Homme"
    confidence = prob if prob > 0.5 else (1 - prob)

    return gender, confidence, prob

def predict_age_v2(interpreter, img_array):
    """Predict age using V2 model"""
    img_input = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    age = int(np.clip(output[0][0], 0, 116))

    return age

def predict_ethnicity_v4(interpreter, img_array):
    """Predict ethnicity using V4 Multitask model (only ethnicity output)"""
    img_input = np.expand_dims(img_array, axis=0)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_input)
    interpreter.invoke()

    # V4 Multitask outputs: [gender, age, ethnicity]
    # We only care about ethnicity (index 2)
    ethnicity_probs = interpreter.get_tensor(output_details[2]['index'])[0]

    idx = np.argmax(ethnicity_probs)
    ethnicity = ETHNICITY_LABELS_V4[idx]
    confidence = ethnicity_probs[idx]

    return ethnicity, confidence, ethnicity_probs

def hybrid_predict(image_path, gender_interp, age_interp, multitask_interp, sizes):
    """Hybrid prediction combining best models for each task"""
    base_size = max(sizes.values())
    img_array = preprocess_image(image_path, base_size)

    # Resize per-model
    gender_img = resize_img(img_array, sizes["gender"])
    age_img = resize_img(img_array, sizes["age"])
    mt_img = resize_img(img_array, sizes["mt"])

    # Gender from V2 (best for gender)
    gender, gender_conf, gender_prob = predict_gender_v2(gender_interp, gender_img)

    # Age from V2
    age = predict_age_v2(age_interp, age_img)

    # Ethnicity from V4 Multitask (best for ethnicity)
    ethnicity, eth_conf, eth_probs = predict_ethnicity_v4(multitask_interp, mt_img)

    return {
        'gender': gender,
        'gender_confidence': gender_conf,
        'gender_prob': gender_prob,
        'age': age,
        'ethnicity': ethnicity,
        'ethnicity_confidence': eth_conf,
        'ethnicity_probs': eth_probs
    }

def main():
    print("=" * 60)
    print("TEST MODE HYBRIDE")
    print("=" * 60)
    print("Genre: V2 Orienté (gender_v2_model.tflite)")
    print("Age: V2 Orienté (age_v2_model.tflite)")
    print("Ethnicité: V4 Multitâche (multitask_model.tflite)")
    print("=" * 60)

    # Load interpreters
    print("\nChargement des modèles...")
    gender_interp = load_interpreter(GENDER_V2_MODEL)
    age_interp = load_interpreter(AGE_V2_MODEL)
    multitask_interp = load_interpreter(MULTITASK_MODEL)
    print("✓ Tous les modèles chargés")

    # Detect input sizes per model
    sizes = {
        "gender": int(gender_interp.get_input_details()[0]["shape"][1]),
        "age": int(age_interp.get_input_details()[0]["shape"][1]),
        "mt": int(multitask_interp.get_input_details()[0]["shape"][1]),
    }
    print(f"Tailles d'entrée détectées: gender {sizes['gender']}, age {sizes['age']}, multitask {sizes['mt']}")

    # Test images (verified paths)
    test_images = [
        (config.IMAGES_DIR / "25_1_0_20170103163054063.jpg.chip.jpg", 25, "Femme", "Blanc"),
        (config.IMAGES_DIR / "35_0_1_20170108224707492.jpg.chip.jpg", 35, "Homme", "Noir"),
        (config.IMAGES_DIR / "45_0_2_20170104174321891.jpg.chip.jpg", 45, "Homme", "Asiatique"),
        (config.IMAGES_DIR / "28_1_3_20170104192939143.jpg.chip.jpg", 28, "Femme", "Indien"),
        (config.IMAGES_DIR / "55_0_0_20170104184424541.jpg.chip.jpg", 55, "Homme", "Blanc"),
    ]

    print("\n" + "=" * 60)
    print("RÉSULTATS HYBRIDES")
    print("=" * 60)

    correct_gender = 0
    correct_ethnicity = 0
    age_errors = []

    for img_path, true_age, true_gender, true_ethnicity in test_images:
        if not os.path.exists(img_path):
            print(f"\n⚠ Image non trouvée: {img_path}")
            continue

        result = hybrid_predict(img_path, gender_interp, age_interp, multitask_interp, sizes)

        # Check correctness
        gender_ok = result['gender'] == true_gender
        ethnicity_ok = result['ethnicity'] == true_ethnicity
        age_error = abs(result['age'] - true_age)

        if gender_ok:
            correct_gender += 1
        if ethnicity_ok:
            correct_ethnicity += 1
        age_errors.append(age_error)

        print(f"\n📷 {os.path.basename(img_path)}")
        print(f"   Réel: {true_age} ans, {true_gender}, {true_ethnicity}")
        print(f"   Prédit HYBRIDE:")
        print(f"   - Genre: {result['gender']} ({result['gender_confidence']*100:.1f}%) {'✓' if gender_ok else '✗'}")
        print(f"   - Age: {result['age']} ans (erreur: {age_error} ans)")
        print(f"   - Ethnicité: {result['ethnicity']} ({result['ethnicity_confidence']*100:.1f}%) {'✓' if ethnicity_ok else '✗'}")

    # Summary
    n_tests = len([t for t in test_images if os.path.exists(t[0])])
    if n_tests > 0:
        print("\n" + "=" * 60)
        print("RÉSUMÉ MODE HYBRIDE")
        print("=" * 60)
        print(f"Genre:     {correct_gender}/{n_tests} correct ({correct_gender/n_tests*100:.0f}%)")
        print(f"Ethnicité: {correct_ethnicity}/{n_tests} correct ({correct_ethnicity/n_tests*100:.0f}%)")
        print(f"Age MAE:   {np.mean(age_errors):.1f} ans")
        print("=" * 60)

        if correct_gender == n_tests and correct_ethnicity == n_tests:
            print("\n🎉 MODE HYBRIDE: 100% correct sur les 5 tests!")
            print("   Combinaison réussie des forces des deux approches!")

if __name__ == "__main__":
    main()
