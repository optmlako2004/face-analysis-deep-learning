"""
Test des 3 modes EfficientNet : Hybride, Orienté V2, Multitâche V4
Charge les modèles TFLite et teste sur des images UTKFace.
"""
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import glob
import time

ASSETS = "FacePredictor/app/src/main/assets"
DATA = "data/UTKFace"
INPUT_SIZE = 128

ETHNICITY_LABELS_V2 = ["Blanc", "Noir", "Asiatique", "Indien", "Autre"]
ETHNICITY_LABELS_V4 = ["Blanc", "Noir", "Asiatique", "Indien"]


def load_image(path):
    img = Image.open(path).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE))
    arr = np.array(img, dtype=np.float32)  # 0-255
    return np.expand_dims(arr, axis=0)


def parse_utkface_filename(path):
    """Parse age_gender_race from UTKFace filename."""
    name = os.path.basename(path).split("_")
    if len(name) >= 3:
        try:
            age = int(name[0])
            gender = int(name[1])  # 0=Male, 1=Female
            race = int(name[2])    # 0=White, 1=Black, 2=Asian, 3=Indian, 4=Others
            return age, gender, race
        except ValueError:
            pass
    return None, None, None


def test_oriented_v2(images, labels):
    """Test Orienté V2 : 3 modèles séparés EfficientNet."""
    print("\n" + "="*60)
    print("MODE: ORIENTÉ V2 (3 modèles EfficientNet séparés)")
    print("="*60)

    gender_model = tf.lite.Interpreter(model_path=f"{ASSETS}/gender_v2_model.tflite")
    gender_model.allocate_tensors()
    age_model = tf.lite.Interpreter(model_path=f"{ASSETS}/age_v2_model.tflite")
    age_model.allocate_tensors()
    eth_model = tf.lite.Interpreter(model_path=f"{ASSETS}/ethnicity_v2_model.tflite")
    eth_model.allocate_tensors()

    correct_gender, correct_eth, total_age_error = 0, 0, 0
    start = time.time()

    for img, (true_age, true_gender, true_race) in zip(images, labels):
        # Gender (with CLAHE-like preprocessing)
        pixels = img[0]
        lum = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]
        min_l, max_l = lum.min(), lum.max()
        r = max_l - min_l
        if r > 0:
            clahe_img = np.clip((pixels - min_l) * (255.0 / r), 0, 255).astype(np.float32)
            clahe_img = np.expand_dims(clahe_img, 0)
        else:
            clahe_img = img

        gender_model.set_tensor(gender_model.get_input_details()[0]['index'], clahe_img)
        gender_model.invoke()
        g_prob = gender_model.get_tensor(gender_model.get_output_details()[0]['index'])[0][0]
        pred_gender = 1 if g_prob > 0.5 else 0

        # Age
        age_model.set_tensor(age_model.get_input_details()[0]['index'], img)
        age_model.invoke()
        pred_age = int(np.clip(age_model.get_tensor(age_model.get_output_details()[0]['index'])[0][0], 0, 116))

        # Ethnicity (5 classes)
        eth_model.set_tensor(eth_model.get_input_details()[0]['index'], img)
        eth_model.invoke()
        eth_probs = eth_model.get_tensor(eth_model.get_output_details()[0]['index'])[0]
        pred_race = np.argmax(eth_probs)

        if pred_gender == true_gender:
            correct_gender += 1
        if pred_race == true_race:
            correct_eth += 1
        total_age_error += abs(pred_age - true_age)

    elapsed = time.time() - start
    n = len(images)
    print(f"  Images testées: {n}")
    print(f"  Genre   - Accuracy: {correct_gender/n*100:.1f}%")
    print(f"  Age     - MAE: {total_age_error/n:.1f} ans")
    print(f"  Ethnie  - Accuracy: {correct_eth/n*100:.1f}%")
    print(f"  Temps total: {elapsed:.2f}s ({elapsed/n*1000:.1f}ms/image)")


def test_multitask_v4(images, labels):
    """Test Multitâche V4 : 1 modèle unifié EfficientNet."""
    print("\n" + "="*60)
    print("MODE: MULTITÂCHE V4 (1 modèle unifié EfficientNet)")
    print("="*60)

    model = tf.lite.Interpreter(model_path=f"{ASSETS}/multitask_model.tflite")
    model.allocate_tensors()

    correct_gender, correct_eth, total_age_error = 0, 0, 0
    start = time.time()

    for img, (true_age, true_gender, true_race) in zip(images, labels):
        model.set_tensor(model.get_input_details()[0]['index'], img)
        model.invoke()

        out_details = model.get_output_details()
        g_prob = model.get_tensor(out_details[0]['index'])[0][0]
        pred_age = int(np.clip(model.get_tensor(out_details[1]['index'])[0][0], 0, 116))
        eth_probs = model.get_tensor(out_details[2]['index'])[0]

        pred_gender = 1 if g_prob > 0.5 else 0
        pred_race = np.argmax(eth_probs)
        # V4 has 4 classes, map Others (4) to closest
        true_race_v4 = min(true_race, 3)

        if pred_gender == true_gender:
            correct_gender += 1
        if pred_race == true_race_v4:
            correct_eth += 1
        total_age_error += abs(pred_age - true_age)

    elapsed = time.time() - start
    n = len(images)
    print(f"  Images testées: {n}")
    print(f"  Genre   - Accuracy: {correct_gender/n*100:.1f}%")
    print(f"  Age     - MAE: {total_age_error/n:.1f} ans")
    print(f"  Ethnie  - Accuracy: {correct_eth/n*100:.1f}%")
    print(f"  Temps total: {elapsed:.2f}s ({elapsed/n*1000:.1f}ms/image)")


def test_hybrid(images, labels):
    """Test Hybride : Gender V2 + Age V2 + Ethnicity V4 Multitask."""
    print("\n" + "="*60)
    print("MODE: HYBRIDE (Gender V2 + Age V2 + Ethnicity V4)")
    print("="*60)

    gender_model = tf.lite.Interpreter(model_path=f"{ASSETS}/gender_v2_model.tflite")
    gender_model.allocate_tensors()
    age_model = tf.lite.Interpreter(model_path=f"{ASSETS}/age_v2_model.tflite")
    age_model.allocate_tensors()
    mt_model = tf.lite.Interpreter(model_path=f"{ASSETS}/multitask_model.tflite")
    mt_model.allocate_tensors()

    correct_gender, correct_eth, total_age_error = 0, 0, 0
    start = time.time()

    for img, (true_age, true_gender, true_race) in zip(images, labels):
        # Gender V2 with CLAHE
        pixels = img[0]
        lum = 0.299 * pixels[:,:,0] + 0.587 * pixels[:,:,1] + 0.114 * pixels[:,:,2]
        min_l, max_l = lum.min(), lum.max()
        r = max_l - min_l
        if r > 0:
            clahe_img = np.clip((pixels - min_l) * (255.0 / r), 0, 255).astype(np.float32)
            clahe_img = np.expand_dims(clahe_img, 0)
        else:
            clahe_img = img

        gender_model.set_tensor(gender_model.get_input_details()[0]['index'], clahe_img)
        gender_model.invoke()
        g_prob = gender_model.get_tensor(gender_model.get_output_details()[0]['index'])[0][0]
        pred_gender = 1 if g_prob > 0.5 else 0

        # Age V2
        age_model.set_tensor(age_model.get_input_details()[0]['index'], img)
        age_model.invoke()
        pred_age = int(np.clip(age_model.get_tensor(age_model.get_output_details()[0]['index'])[0][0], 0, 116))

        # Ethnicity from V4 Multitask
        mt_model.set_tensor(mt_model.get_input_details()[0]['index'], img)
        mt_model.invoke()
        out_details = mt_model.get_output_details()
        eth_probs = mt_model.get_tensor(out_details[2]['index'])[0]
        pred_race = np.argmax(eth_probs)
        true_race_v4 = min(true_race, 3)

        if pred_gender == true_gender:
            correct_gender += 1
        if pred_race == true_race_v4:
            correct_eth += 1
        total_age_error += abs(pred_age - true_age)

    elapsed = time.time() - start
    n = len(images)
    print(f"  Images testées: {n}")
    print(f"  Genre   - Accuracy: {correct_gender/n*100:.1f}%")
    print(f"  Age     - MAE: {total_age_error/n:.1f} ans")
    print(f"  Ethnie  - Accuracy: {correct_eth/n*100:.1f}%")
    print(f"  Temps total: {elapsed:.2f}s ({elapsed/n*1000:.1f}ms/image)")


if __name__ == "__main__":
    print("Chargement des images de test UTKFace...")

    all_files = sorted(glob.glob(f"{DATA}/*.jpg*"))
    np.random.seed(42)
    sample_indices = np.random.choice(len(all_files), min(200, len(all_files)), replace=False)
    sample_files = [all_files[i] for i in sample_indices]

    images = []
    labels = []
    for f in sample_files:
        age, gender, race = parse_utkface_filename(f)
        if age is not None:
            images.append(load_image(f))
            labels.append((age, gender, race))

    print(f"{len(images)} images chargées")

    test_hybrid(images, labels)
    test_multitask_v4(images, labels)
    test_oriented_v2(images, labels)

    print("\n" + "="*60)
    print("TESTS TERMINÉS")
    print("="*60)
