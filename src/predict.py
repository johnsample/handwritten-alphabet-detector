
from pathlib import Path
import sys

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "alphabet_cnn.keras"

IMAGE_SIZE = 28
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ------------------------------------------------------------
# 2. IMAGE PREPROCESSING
# Keep this consistent with train_model.py
# ------------------------------------------------------------

def preprocess_image(image_path):

    image = Image.open(image_path).convert("L")
    img = np.array(image, dtype=np.uint8)

    mean_value = img.mean()

    if mean_value > 127:
        mask = img < 200
    else:
        mask = img > 55

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():

        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        result = np.array(image, dtype=np.float32) / 255.0

        return np.expand_dims(result, axis=-1)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    cropped = image.crop(
        (x_min, y_min, x_max + 1, y_max + 1)
    )

    width, height = cropped.size
    padding = int(max(width, height) * 0.20)

    cropped = ImageOps.expand(
        cropped,
        border=padding,
        fill=255
    )

    width, height = cropped.size
    scale = 20 / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    cropped = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    canvas = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        255
    )

    x_offset = (IMAGE_SIZE - new_width) // 2
    y_offset = (IMAGE_SIZE - new_height) // 2

    canvas.paste(cropped, (x_offset, y_offset))

    result = np.array(canvas, dtype=np.float32)

    result = (255.0 - result) / 255.0

    result = np.expand_dims(result, axis=-1)

    return result


# ------------------------------------------------------------
# 3. PREDICT LETTER
# ------------------------------------------------------------

def predict_letter(image_path):

    if not MODEL_PATH.exists():
        print("ERROR: Trained model was not found.")
        print(f"Expected location: {MODEL_PATH}")
        return

    if not Path(image_path).exists():
        print(f"ERROR: Image not found: {image_path}")
        return

    model = tf.keras.models.load_model(MODEL_PATH)

    processed = preprocess_image(image_path)

    input_image = np.expand_dims(processed, axis=0)

    probabilities = model.predict(
        input_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))
    predicted_letter = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index]) * 100

    print("\n" + "=" * 40)
    print("HANDWRITTEN ALPHABET PREDICTION")
    print("=" * 40)
    print(f"Image: {image_path}")
    print(f"Predicted Letter: {predicted_letter}")
    print(f"Confidence: {confidence:.2f}%")
    print("=" * 40)


# ------------------------------------------------------------
# 4. RUN FROM COMMAND LINE
# ------------------------------------------------------------

if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print("python src/predict.py path_to_image.png")
    else:
        predict_letter(sys.argv[1])