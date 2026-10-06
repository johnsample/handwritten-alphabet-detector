import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = ROOT / "models" / "alphabet_cnn.keras"

IMAGE_SIZE = 28
CANVAS_SIZE = 280

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


# ============================================================
# 2. LOAD TRAINED MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Trained model was not found:\n{MODEL_PATH}"
    )

model = tf.keras.models.load_model(MODEL_PATH)


# ============================================================
# 3. CREATE MAIN WINDOW
# ============================================================

root = tk.Tk()

root.title("Handwritten Alphabet Detector")

root.geometry("500x650")

root.resizable(False, False)


# ============================================================
# 4. TITLE
# ============================================================

title_label = tk.Label(
    root,
    text="HANDWRITTEN ALPHABET DETECTOR",
    font=("Arial", 20, "bold")
)

title_label.pack(pady=15)


instruction_label = tk.Label(
    root,
    text="Draw an English alphabet letter A-Z",
    font=("Arial", 12)
)

instruction_label.pack(pady=5)


# ============================================================
# 5. CANVAS
# ============================================================

canvas = tk.Canvas(
    root,
    width=CANVAS_SIZE,
    height=CANVAS_SIZE,
    bg="white",
    highlightthickness=2
)

canvas.pack(pady=15)


# PIL image stores the drawing
drawing_image = Image.new(
    "L",
    (CANVAS_SIZE, CANVAS_SIZE),
    255
)

drawing = ImageDraw.Draw(drawing_image)


# ============================================================
# 6. MOUSE DRAWING
# ============================================================

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


canvas.bind(
    "<Button-1>",
    start_drawing
)

canvas.bind(
    "<B1-Motion>",
    draw
)

canvas.bind(
    "<ButtonRelease-1>",
    stop_drawing
)


# ============================================================
# 7. PREPROCESS IMAGE
# ============================================================

def preprocess_image(image):

    image = image.convert("L")

    img = np.array(
        image,
        dtype=np.uint8
    )

    mask = img < 200

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    # If image is completely blank
    if not rows.any() or not cols.any():

        return None

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    cropped = image.crop(
        (
            x_min,
            y_min,
            x_max + 1,
            y_max + 1
        )
    )

    # Add padding
    width, height = cropped.size

    padding = int(
        max(width, height) * 0.20
    )

    cropped = ImageOps.expand(
        cropped,
        border=padding,
        fill=255
    )

    # Resize while preserving shape
    width, height = cropped.size

    scale = 20 / max(
        width,
        height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    cropped = cropped.resize(
        (
            new_width,
            new_height
        ),
        Image.Resampling.LANCZOS
    )

    # Create 28x28 canvas
    canvas_image = Image.new(
        "L",
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        255
    )

    x_offset = (
        IMAGE_SIZE - new_width
    ) // 2

    y_offset = (
        IMAGE_SIZE - new_height
    ) // 2

    canvas_image.paste(
        cropped,
        (
            x_offset,
            y_offset
        )
    )

    # Convert white background / black letter
    # into normalized CNN format
    result = np.array(
        canvas_image,
        dtype=np.float32
    )

    result = (
        255.0 - result
    ) / 255.0

    result = np.expand_dims(
        result,
        axis=-1
    )

    return result


# ============================================================
# 8. DISPLAY PREDICTION
# ============================================================

def show_prediction(probabilities):

    predicted_index = int(
        np.argmax(probabilities)
    )

    predicted_letter = CLASS_NAMES[
        predicted_index
    ]

    confidence = (
        float(
            probabilities[
                predicted_index
            ]
        ) * 100
    )

    # Top 3 predictions
    top_indices = np.argsort(
        probabilities
    )[-3:][::-1]

    top_text = ""

    for index in top_indices:

        letter = CLASS_NAMES[index]

        percentage = (
            float(
                probabilities[index]
            ) * 100
        )

        top_text += (
            f"{letter}: "
            f"{percentage:.2f}%\n"
        )

    result_label.config(
        text=(
            f"Prediction: {predicted_letter}\n"
            f"Confidence: {confidence:.2f}%"
        )
    )

    top_label.config(
        text=(
            "Top 3 Predictions\n"
            + top_text
        )
    )


# ============================================================
# 9. PREDICT DRAWING
# ============================================================

def predict_drawing():

    processed = preprocess_image(
        drawing_image
    )

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

    show_prediction(
        probabilities
    )


# ============================================================
# 10. CLEAR DRAWING
# ============================================================

def clear_canvas():

    canvas.delete("all")

    drawing_image.paste(
        255,
        (
            0,
            0,
            CANVAS_SIZE,
            CANVAS_SIZE
        )
    )

    result_label.config(
        text=(
            "Prediction: -\n"
            "Confidence: -"
        )
    )

    top_label.config(
        text="Top 3 Predictions\n-"
    )


# ============================================================
# 11. SAVE DRAWING
# ============================================================

def save_drawing():

    file_path = filedialog.asksaveasfilename(
        title="Save Drawing",
        defaultextension=".png",
        filetypes=[
            (
                "PNG Image",
                "*.png"
            ),
            (
                "JPEG Image",
                "*.jpg"
            )
        ]
    )

    if not file_path:
        return

    drawing_image.save(
        file_path
    )

    messagebox.showinfo(
        "Saved",
        f"Drawing saved successfully:\n\n{file_path}"
    )


# ============================================================
# 12. UPLOAD IMAGE
# ============================================================

def upload_image():

    file_path = filedialog.askopenfilename(
        title="Select Alphabet Image",
        filetypes=[
            (
                "Image Files",
                "*.png *.jpg *.jpeg *.bmp"
            )
        ]
    )

    if not file_path:
        return

    try:

        image = Image.open(
            file_path
        ).convert("L")

        processed = preprocess_image(
            image
        )

        if processed is None:

            messagebox.showwarning(
                "Invalid Image",
                "The selected image appears to be blank."
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

        show_prediction(
            probabilities
        )

    except Exception as error:

        messagebox.showerror(
            "Error",
            f"Could not process the image:\n\n{error}"
        )


# ============================================================
# 13. BUTTON FRAME
# ============================================================

button_frame = tk.Frame(
    root
)

button_frame.pack(
    pady=10
)


# Predict button
predict_button = tk.Button(
    button_frame,
    text="Predict",
    font=("Arial", 12, "bold"),
    width=12,
    command=predict_drawing
)

predict_button.grid(
    row=0,
    column=0,
    padx=5
)


# Clear button
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
    padx=5
)


# Upload button
upload_button = tk.Button(
    button_frame,
    text="Upload Image",
    font=("Arial", 12, "bold"),
    width=12,
    command=upload_image
)

upload_button.grid(
    row=1,
    column=0,
    padx=5,
    pady=8
)


# Save button
save_button = tk.Button(
    button_frame,
    text="Save Drawing",
    font=("Arial", 12, "bold"),
    width=12,
    command=save_drawing
)

save_button.grid(
    row=1,
    column=1,
    padx=5,
    pady=8
)


# ============================================================
# 14. RESULT LABEL
# ============================================================

result_label = tk.Label(
    root,
    text=(
        "Prediction: -\n"
        "Confidence: -"
    ),
    font=("Arial", 16, "bold")
)

result_label.pack(
    pady=10
)


# ============================================================
# 15. TOP 3 PREDICTIONS
# ============================================================

top_label = tk.Label(
    root,
    text="Top 3 Predictions\n-",
    font=("Arial", 12)
)

top_label.pack(
    pady=5
)


# ============================================================
# 16. START APPLICATION
# ============================================================

root.mainloop()