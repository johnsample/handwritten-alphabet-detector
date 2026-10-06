import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw


# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "alphabet_cnn.keras"

IMAGE_SIZE = 28
CANVAS_SIZE = 280

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ------------------------------------------------------------
# 2. LOAD TRAINED MODEL
# ------------------------------------------------------------

if not MODEL_PATH.exists():
    messagebox.showerror(
        "Model Error",
        f"Trained model was not found.\n\nExpected:\n{MODEL_PATH}"
    )
    raise FileNotFoundError(MODEL_PATH)

model = tf.keras.models.load_model(MODEL_PATH)


# ------------------------------------------------------------
# 3. CREATE WINDOW
# ------------------------------------------------------------

root = tk.Tk()

root.title("Handwritten Alphabet Detector")
root.geometry("420x600")
root.resizable(False, False)


# ------------------------------------------------------------
# 4. TITLE
# ------------------------------------------------------------

title_label = tk.Label(
    root,
    text="HANDWRITTEN ALPHABET DETECTOR",
    font=("Arial", 18, "bold")
)

title_label.pack(pady=15)


instruction_label = tk.Label(
    root,
    text="Draw an English letter A-Z below",
    font=("Arial", 12)
)

instruction_label.pack(pady=5)


# ------------------------------------------------------------
# 5. DRAWING CANVAS
# ------------------------------------------------------------

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="white",
    highlightthickness=2
)

canvas.pack(pady=10)


# PIL image used to store the drawing
drawing_image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    255
)

drawing = ImageDraw.Draw(drawing_image)


# ------------------------------------------------------------
# 6. DRAW WITH MOUSE
# ------------------------------------------------------------

last_x = None
last_y = None


def start_drawing(event):
    global last_x, last_y

    last_x = event.x
    last_y = event.y


def draw(event):
    global last_x, last_y

    if last_x is not None and last_y is not None:

        canvas.create_line(
            last_x,
            last_y,
            event.x,
            event.y,
            fill="black",
            width=18,
            capstyle=tk.ROUND,
            smooth=True
        )

        drawing.line(
            (last_x, last_y, event.x, event.y),
            fill=0,
            width=18
        )

    last_x = event.x
    last_y = event.y


def stop_drawing(event):
    global last_x, last_y

    last_x = None
    last_y = None


canvas.bind("<Button-1>", start_drawing)
canvas.bind("<B1-Motion>", draw)
canvas.bind("<ButtonRelease-1>", stop_drawing)


# ------------------------------------------------------------
# 7. PREPARE IMAGE FOR CNN
# ------------------------------------------------------------

def prepare_drawing():

    image = drawing_image.copy()

    # Find bounding box of the letter
    array = np.array(image)

    mask = array < 200

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    cropped = image.crop(
        (x_min, y_min, x_max + 1, y_max + 1)
    )

    # Add padding
    width, height = cropped.size

    padding = int(max(width, height) * 0.20)

    padded = Image.new(
        "L",
        (
            width + padding * 2,
            height + padding * 2
        ),
        255
    )

    padded.paste(
        cropped,
        (padding, padding)
    )

    # Resize while maintaining proportions
    width, height = padded.size

    scale = 20 / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    resized = padded.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 white canvas
    final_image = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        255
    )

    x_offset = (IMAGE_SIZE - new_width) // 2
    y_offset = (IMAGE_SIZE - new_height) // 2

    final_image.paste(
        resized,
        (x_offset, y_offset)
    )

    # Convert black/white image to CNN format
    result = np.array(
        final_image,
        dtype=np.float32
    )

    result = (255.0 - result) / 255.0

    result = np.expand_dims(
        result,
        axis=-1
    )

    return result


# ------------------------------------------------------------
# 8. PREDICT LETTER
# ------------------------------------------------------------

def predict_letter():

    processed = prepare_drawing()

    if processed is None:

        messagebox.showwarning(
            "No Drawing",
            "Please draw a letter first."
        )

        return

    input_image = np.expand_dims(
        processed,
        axis=0
    )

    probabilities = model.predict(
        input_image,
        verbose=0
    )[0]

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_letter = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        float(probabilities[predicted_index])
        * 100
    )

    result_label.config(
        text=f"Prediction: {predicted_letter}\n"
             f"Confidence: {confidence:.2f}%"
    )


# ------------------------------------------------------------
# 9. CLEAR CANVAS
# ------------------------------------------------------------

def clear_canvas():

    canvas.delete("all")

    drawing_image.paste(
        255,
        [0, 0, CANVAS_SIZE, CANVAS_SIZE]
    )

    result_label.config(
        text="Prediction: -\nConfidence: -"
    )


# ------------------------------------------------------------
# 10. BUTTONS
# ------------------------------------------------------------

button_frame = tk.Frame(root)

button_frame.pack(pady=10)


predict_button = tk.Button(
    button_frame,
    text="Predict",
    font=("Arial", 12, "bold"),
    width=12,
    command=predict_letter
)

predict_button.grid(
    row=0,
    column=0,
    padx=10
)


clear_button = tk.Button(
    button_frame,
    text="Clear",
    font=("Arial", 12, "bold"),
    width=12,
    command=clear_canvas
)

clear_button.grid(
    row=0,
    column=1,
    padx=10
)


# ------------------------------------------------------------
# 11. RESULT
# ------------------------------------------------------------

result_label = tk.Label(
    root,
    text="Prediction: -\nConfidence: -",
    font=("Arial", 16, "bold")
)

result_label.pack(pady=15)


# ------------------------------------------------------------
# 12. START APPLICATION
# ------------------------------------------------------------

root.mainloop()