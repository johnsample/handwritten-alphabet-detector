# HANDWRITTEN ALPHABET DETECTOR

## Deep Learning Project – Image Classification Using Convolutional Neural Networks (CNN)

A deep learning-based application that recognizes handwritten English alphabets from **A to Z** using a Convolutional Neural Network (CNN).

The project includes image preprocessing, CNN model training, evaluation, individual image prediction, and an interactive **Tkinter graphical user interface** that allows users to draw or upload handwritten alphabet images.

---

## Project Overview

Handwritten character recognition is an important application of Artificial Intelligence, Machine Learning, and Computer Vision.

This project develops a CNN-based system capable of classifying handwritten English alphabets into **26 classes (A–Z)**.

The system processes an input image by converting it to grayscale, detecting the character region, cropping unnecessary background, applying padding, resizing the image to **28 × 28 pixels**, and normalizing pixel values before sending it to the trained CNN.

The application can then display the predicted alphabet, confidence score, and top prediction results.

---

## Features

* Recognition of handwritten English alphabets A–Z
* 26-class image classification
* CNN-based deep learning model
* Grayscale image preprocessing
* Character detection and cropping
* Image padding and resizing
* Pixel normalization
* Training and validation monitoring
* Test dataset evaluation
* Accuracy and loss graphs
* Confusion matrix
* Classification report
* Individual image prediction
* Prediction confidence score
* Top-3 predictions
* Tkinter drawing interface
* Image upload functionality
* Save drawing functionality
* Saved Keras model for future predictions

---

## Dataset

### Dataset Name

**Handwritten English Characters and Digits Dataset**

The project uses handwritten English alphabet images representing the 26 classes:

```text
A B C D E F G H I J K L M
N O P Q R S T U V W X Y Z
```

### Dataset Structure

```text
dataset/
└── archive/
    └── handwritten-english-characters-and-digits/
        └── combined_folder/
            ├── train/
            │   ├── A_caps/
            │   ├── B_caps/
            │   ├── ...
            │   └── Z_caps/
            │
            └── test/
                ├── A_caps/
                ├── B_caps/
                ├── ...
                └── Z_caps/
```

The dataset contains:

* Training images: 1,144
* Training samples: 972
* Validation samples: 172
* Test images: 286
* Classes: 26
* Images per test class: 11

---

## Image Preprocessing

The preprocessing pipeline is:

```text
Input Image
     ↓
Grayscale Conversion
     ↓
Character Detection
     ↓
Cropping
     ↓
Padding
     ↓
Resize to 28 × 28
     ↓
Normalization
     ↓
28 × 28 × 1 Input
     ↓
CNN Model
```

Pixel values are normalized using:

```text
Normalized Pixel Value = Pixel Value / 255
```

The final CNN input format is:

```text
28 × 28 × 1
```

---

## CNN Architecture

The CNN architecture consists of:

```text
Input: 28 × 28 × 1
        ↓
Conv2D – 32 filters
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D – 64 filters
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Conv2D – 128 filters
        ↓
Batch Normalization
        ↓
Max Pooling
        ↓
Flatten
        ↓
Dense – 128 neurons
        ↓
Dropout
        ↓
Dense – 26 neurons
        ↓
Softmax
        ↓
Predicted Alphabet
```

### Model Parameters

* Total parameters: 244,506
* Trainable parameters: 244,058
* Non-trainable parameters: 448

---

## Model Performance

The reported model performance is:

| Metric              | Result |
| ------------------- | -----: |
| Number of Classes   |     26 |
| Training Accuracy   | 99.90% |
| Validation Accuracy | 94.77% |
| Test Accuracy       | 96.15% |
| Test Loss           | 0.1627 |
| Test Images         |    286 |
| Completed Epochs    |     28 |

During individual prediction testing, the sample image was classified as:

```text
Predicted Letter: A
Confidence: 99.82%
```

---

## Technologies Used

* Python
* TensorFlow
* Keras
* NumPy
* Scikit-learn
* Pillow
* Matplotlib
* Tkinter
* Visual Studio Code
* Windows

---

## Project Structure

```text
Handwritten_Alphabet_Detector/
│
├── dataset/
│
├── models/
│   └── alphabet_cnn.keras
│
├── src/
│   ├── train_model.py
│   ├── predict.py
│   ├── app.py
│   └── app_backup.py
│
├── outputs/
│   ├── accuracy_plot.png
│   ├── loss_plot.png
│   ├── confusion_matrix.png
│   ├── classification_report.txt
│   └── metrics.json
│
├── report/
│   └── Handwritten_Alphabet_Detector_Report.pdf
│
├── screenshots/
│
├── README.md
└── requirements.txt
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/johnsample/handwritten-alphabet-detector.git
```

Move into the project directory:

```bash
cd handwritten-alphabet-detector
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the virtual environment on Windows:

```bash
.venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Train the CNN

```bash
python src/train_model.py
```

### Predict an Image

```bash
python src/predict.py
```

### Launch the Tkinter Application

```bash
python src/app.py
```

The GUI allows the user to:

1. Draw an alphabet using the mouse.
2. Click **Predict**.
3. View the predicted alphabet.
4. View the confidence score.
5. View top prediction results.
6. Clear the drawing.
7. Upload an existing image.
8. Save the drawing.

---

## Model File

The trained model is saved as:

```text
models/alphabet_cnn.keras
```

The saved model allows predictions to be performed without retraining the CNN.

---

## Evaluation

The project evaluates the CNN using:

* Accuracy
* Loss
* Classification report
* Confusion matrix
* Individual image prediction

Training and validation graphs are also generated to visualize model performance.

---

## Applications

The project can be useful as a basic foundation for:

* Handwritten form digitization
* Educational applications
* Document recognition
* Character recognition systems
* Computer vision applications
* AI-based handwriting recognition

---

## Limitations

* The system recognizes only English alphabets A–Z.
* It is primarily designed for individual character recognition.
* Very unclear or noisy images may reduce prediction accuracy.
* Handwriting styles significantly different from the training data may be harder to classify.
* Performance may vary on completely different datasets.

---

## Future Enhancements

Possible future improvements include:

* Word and sentence recognition
* Larger and more diverse datasets
* Improved CNN architectures
* Additional data augmentation
* Web application deployment
* Mobile application integration
* Real-time camera-based recognition
* Digit and special-character recognition
* Model optimization and quantization
* Enhanced prediction history and visualization

---

## Author

**John Joshua J**

**Project:** Handwritten Alphabet Detector

**Type:** Deep Learning Project

**Date:** September 2026

---

## License

This project was developed as an academic deep learning project.
