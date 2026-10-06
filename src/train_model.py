from pathlib import Path
from sklearn.model_selection import train_test_split
import json

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATASET_DIR = (
    ROOT
    / "dataset"
    / "archive"
    / "handwritten-english-characters-and-digits"
    / "combined_folder"
)

TRAIN_DIR = DATASET_DIR / "train"
TEST_DIR = DATASET_DIR / "test"

MODEL_DIR = ROOT / "models"
OUTPUT_DIR = ROOT / "outputs"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. SETTINGS
# ============================================================

IMAGE_SIZE = 28
BATCH_SIZE = 32
EPOCHS = 30

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
NUM_CLASSES = len(CLASS_NAMES)


# ============================================================
# 3. IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image_path):
    """
    Convert a large RGB image into a centered 28x28 grayscale image.

    Steps:
    1. Open image
    2. Convert to grayscale
    3. Detect foreground
    4. Crop around the handwritten character
    5. Preserve aspect ratio
    6. Center character on 28x28 canvas
    7. Normalize pixels
    """

    image = Image.open(image_path).convert("L")

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    img = np.array(image, dtype=np.uint8)

    # --------------------------------------------------------
    # Automatically determine whether background is white
    # or dark.
    # --------------------------------------------------------

    mean_value = img.mean()

    if mean_value > 127:
        # White background / dark character
        threshold = 200
        mask = img < threshold
    else:
        # Dark background / light character
        threshold = 55
        mask = img > threshold

    # --------------------------------------------------------
    # Find character boundaries
    # --------------------------------------------------------

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not rows.any() or not cols.any():
        # If no foreground is detected,
        # simply resize the original image.
        image = image.resize(
            (IMAGE_SIZE, IMAGE_SIZE),
            Image.Resampling.LANCZOS
        )

        result = np.array(image, dtype=np.float32) / 255.0

        return np.expand_dims(result, axis=-1)

    # Bounding box
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Crop character
    cropped = image.crop(
        (
            x_min,
            y_min,
            x_max + 1,
            y_max + 1
        )
    )

    # --------------------------------------------------------
    # Add padding
    # --------------------------------------------------------

    width, height = cropped.size

    padding = int(max(width, height) * 0.20)

    cropped = ImageOps.expand(
        cropped,
        border=padding,
        fill=255
    )

    # --------------------------------------------------------
    # Resize while keeping aspect ratio
    # --------------------------------------------------------

    width, height = cropped.size

    scale = 20 / max(width, height)

    new_width = max(1, int(width * scale))
    new_height = max(1, int(height * scale))

    cropped = cropped.resize(
        (new_width, new_height),
        Image.Resampling.LANCZOS
    )

    # --------------------------------------------------------
    # Create 28x28 white canvas
    # --------------------------------------------------------

    canvas = Image.new(
        "L",
        (IMAGE_SIZE, IMAGE_SIZE),
        255
    )

    # Center character
    x_offset = (IMAGE_SIZE - new_width) // 2
    y_offset = (IMAGE_SIZE - new_height) // 2

    canvas.paste(
        cropped,
        (x_offset, y_offset)
    )

    # --------------------------------------------------------
    # Convert to numpy
    # --------------------------------------------------------

    result = np.array(
        canvas,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Invert if necessary
    #
    # CNN will use:
    # black background = 0
    # white character = 1
    # --------------------------------------------------------

    result = 255.0 - result

    result = result / 255.0

    # Add channel dimension
    result = np.expand_dims(
        result,
        axis=-1
    )

    return result


# ============================================================
# 4. LOAD DATASET
# ============================================================

def load_images(folder_path):

    images = []
    labels = []

    print(f"\nLoading images from: {folder_path}")

    for label, letter in enumerate(CLASS_NAMES):

        class_folder = folder_path / f"{letter}_caps"

        if not class_folder.exists():
            print(
                f"WARNING: Missing folder: "
                f"{class_folder}"
            )
            continue

        image_files = []

        for extension in [
            "*.png",
            "*.jpg",
            "*.jpeg",
            "*.bmp"
        ]:
            image_files.extend(
                class_folder.rglob(extension)
            )

        print(
            f"{letter}: "
            f"{len(image_files)} images"
        )

        for image_file in image_files:

            try:

                processed = preprocess_image(
                    image_file
                )

                images.append(processed)
                labels.append(label)

            except Exception as error:

                print(
                    f"Could not process "
                    f"{image_file}: {error}"
                )

    return (
        np.array(images, dtype=np.float32),
        np.array(labels, dtype=np.int32)
    )


# ============================================================
# 5. CHECK DATASET
# ============================================================

print("=" * 60)
print("HANDWRITTEN ALPHABET DETECTOR")
print("=" * 60)

print("\nDataset directory:")
print(DATASET_DIR)

print("\nTraining directory:")
print(TRAIN_DIR)

print("\nTesting directory:")
print(TEST_DIR)

if not TRAIN_DIR.exists():

    print(
        "\nERROR: Training directory "
        "does not exist."
    )

    raise SystemExit


if not TEST_DIR.exists():

    print(
        "\nERROR: Testing directory "
        "does not exist."
    )

    raise SystemExit


# ============================================================
# 6. LOAD DATA
# ============================================================

X_train, y_train = load_images(
    TRAIN_DIR
)

X_test, y_test = load_images(
    TEST_DIR
)


print("\n" + "=" * 60)
print("DATASET SUMMARY")
print("=" * 60)

print(
    f"Training images: {len(X_train)}"
)

print(
    f"Testing images:  {len(X_test)}"
)


if len(X_train) == 0:

    print(
        "\nERROR: No training images found."
    )

    raise SystemExit


if len(X_test) == 0:

    print(
        "\nERROR: No testing images found."
    )

    raise SystemExit



# ============================================================
# 7. DISPLAY TRAINING AND TEST SAMPLES
# ============================================================

fig, axes = plt.subplots(2, 10, figsize=(15, 4))

# Show 10 training samples
for i in range(min(10, len(X_train))):
    axes[0, i].imshow(
        X_train[i].squeeze(),
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[0, i].set_title(
        f"Train: {CLASS_NAMES[y_train[i]]}"
    )
    axes[0, i].axis("off")

# Show 10 test samples
for i in range(min(10, len(X_test))):
    axes[1, i].imshow(
        X_test[i].squeeze(),
        cmap="gray",
        vmin=0,
        vmax=1
    )
    axes[1, i].set_title(
        f"Test: {CLASS_NAMES[y_test[i]]}"
    )
    axes[1, i].axis("off")

plt.tight_layout()

sample_path = OUTPUT_DIR / "preprocessed_samples.png"

plt.savefig(sample_path, dpi=150)
plt.close()

print(f"\nTraining and test samples saved to:\n{sample_path}")


# ============================================================
# 8. BUILD CNN
# ============================================================

print("\n" + "=" * 60)
print("BUILDING CNN MODEL")
print("=" * 60)

model = models.Sequential([

    layers.Input(
        shape=(
            IMAGE_SIZE,
            IMAGE_SIZE,
            1
        )
    ),

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Third convolution block
    layers.Conv2D(
        128,
        (3, 3),
        padding="same",
        activation="relu"
    ),

    layers.BatchNormalization(),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Classifier
    layers.Flatten(),

    layers.Dense(
        128,
        activation="relu"
    ),

    layers.Dropout(
        0.3
    ),

    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )
])


# ============================================================
# 9. COMPILE MODEL
# ============================================================

model.compile(

    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001
    ),

    loss="sparse_categorical_crossentropy",

    metrics=[
        "accuracy"
    ]
)


print("\nModel architecture:")

model.summary()


# ============================================================
# 10. CALLBACKS
# ============================================================

early_stopping = callbacks.EarlyStopping(

    monitor="val_accuracy",

    patience=7,

    mode="max",

    restore_best_weights=True
)


reduce_learning_rate = (
    callbacks.ReduceLROnPlateau(

        monitor="val_loss",

        factor=0.5,

        patience=3,

        min_lr=1e-6
    )
)


# ============================================================
# 11. TRAIN MODEL
# ============================================================

print("\n" + "=" * 60)
print("STARTING TRAINING")
print("=" * 60)

# Create a stratified validation set.
# This helps ensure every letter is represented in both sets.
stratify_labels = (
    np.argmax(y_train, axis=1)
    if y_train.ndim > 1
    else y_train
)

X_train_part, X_val, y_train_part, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.15,
    random_state=42,
    shuffle=True,
    stratify=stratify_labels
)

print("Training samples:", len(X_train_part))
print("Validation samples:", len(X_val))

history = model.fit(
    X_train_part,
    y_train_part,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[
        early_stopping,
        reduce_learning_rate
    ],
    shuffle=True,
    verbose=1
)

# ============================================================
# 12. EVALUATE
# ============================================================

print("\n" + "=" * 60)
print("EVALUATING MODEL")
print("=" * 60)

test_loss, test_accuracy = (
    model.evaluate(
        X_test,
        y_test,
        verbose=1
    )
)


print(
    f"\nTest Loss: "
    f"{test_loss:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy:.4f}"
)

print(
    f"Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)


# ============================================================
# 13. PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)

predictions = model.predict(
    X_test,
    batch_size=BATCH_SIZE,
    verbose=1
)

y_pred = np.argmax(
    predictions,
    axis=1
)


# ============================================================
# 14. CLASSIFICATION REPORT
# ============================================================

report = classification_report(

    y_test,

    y_pred,

    labels=list(
        range(NUM_CLASSES)
    ),

    target_names=CLASS_NAMES,

    zero_division=0
)


print(
    "\n" + "=" * 60
)

print(
    "CLASSIFICATION REPORT"
)

print(
    "=" * 60
)

print(report)


report_path = (
    OUTPUT_DIR /
    "classification_report.txt"
)

with open(
    report_path,
    "w",
    encoding="utf-8"
) as file:

    file.write(report)


# ============================================================
# 15. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=list(
        range(NUM_CLASSES)
    )
)


fig, ax = plt.subplots(
    figsize=(12, 10)
)

disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=CLASS_NAMES
)

disp.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title(
    "Confusion Matrix - "
    "Handwritten Alphabet Detector"
)

plt.tight_layout()


confusion_path = (
    OUTPUT_DIR /
    "confusion_matrix.png"
)

plt.savefig(
    confusion_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 16. ACCURACY GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.title(
    "Training and Validation Accuracy"
)

plt.xlabel("Epoch")

plt.ylabel("Accuracy")

plt.legend()

plt.grid(True)


accuracy_path = (
    OUTPUT_DIR /
    "accuracy_plot.png"
)

plt.savefig(
    accuracy_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 17. LOSS GRAPH
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.plot(
    history.history["loss"],
    label="Training Loss"
)

plt.plot(
    history.history["val_loss"],
    label="Validation Loss"
)

plt.title(
    "Training and Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.legend()

plt.grid(True)


loss_path = (
    OUTPUT_DIR /
    "loss_plot.png"
)

plt.savefig(
    loss_path,
    dpi=150,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# 18. SAVE MODEL
# ============================================================

model_path = (
    MODEL_DIR /
    "alphabet_cnn.keras"
)

model.save(
    model_path
)


# ============================================================
# 19. SAVE METRICS
# ============================================================

metrics = {

    "test_loss":
        float(test_loss),

    "test_accuracy":
        float(test_accuracy),

    "test_accuracy_percent":
        float(test_accuracy * 100),

    "training_images":
        int(len(X_train)),

    "testing_images":
        int(len(X_test)),

    "image_size":
        IMAGE_SIZE,

    "batch_size":
        BATCH_SIZE,

    "epochs_requested":
        EPOCHS,

    "epochs_completed":
        len(
            history.history["loss"]
        ),

    "classes":
        CLASS_NAMES
}


metrics_path = (
    OUTPUT_DIR /
    "metrics.json"
)

with open(
    metrics_path,
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        metrics,
        file,
        indent=4
    )


# ============================================================
# 20. FINAL OUTPUT
# ============================================================

print("\n" + "=" * 60)
print("TRAINING COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    f"\nFinal Test Accuracy: "
    f"{test_accuracy * 100:.2f}%"
)

print("\nModel:")
print(model_path)

print("\nOutput files:")
print(sample_path)
print(accuracy_path)
print(loss_path)
print(confusion_path)
print(report_path)
print(metrics_path)

print("\nNext step: prediction program.")