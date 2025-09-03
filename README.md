
# Fashion MNIST Classification Project

## 📋 Project Overview
This project implements a Convolutional Neural Network (CNN) with **6 layers** to classify images from the Fashion MNIST dataset.  
The implementation is provided in both **Python** and **R**, demonstrating model construction, training, and prediction.

---

## Assignment Requirements Met
-  CNN with **6 layers** implemented in both Python and R  
-  **Fashion MNIST dataset** used for classification  
-  **Predictions on two sample images** with probability outputs  
-  **Keras/TensorFlow** used for deep learning  
-  **Visualization** of predictions saved as PNG files  
-  **Logging** of output to timestamped text files  
-  **Custom class-like structure** in both Python and R  

---

##  Project Structure
fashion-mnist-classification/
│
├── fashion_mnist_cnn.py # Python implementation
├── fashion_mnist_cnn.R # R implementation
├── python_output_YYYYMMDD_HHMMSS.txt # Python output log (generated)
├── r_output_YYYYMMDD_HHMMSS.txt # R output log (generated)
├── predictions.png # Prediction visualization (generated)
└── README.md # This file



## Requirements
- TensorFlow / Keras
- R with keras library (optional)

# Create a virtual environment
- python3 -m venv fashion_env

# Activate the virtual environment
- source fashion_env/bin/activate

# Now install the required packages
- pip install tensorflow numpy matplotlib

# Verify installation
- python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"

## How to Run (Python)
1. Install dependencies:
   pip install tensorflow keras matplotlib numpy
2. Run the script:
   python fashion_mnist_cnn.py

## How to Run (R)
1. Install keras package in R.

2. Run:
   Rscript fashion_mnist_cnn.R

## Predictions
The model predicts at least two images from the Fashion MNIST test set, showing predicted vs. true labels.

## Files
- `fashion_mnist_cnn.py` → Python code
- `fashion_mnist_cnn.R` → R code
- `README.md` → Instructions
# Fashion MNIST Classification with CNN

## Overview
This project implements a Convolutional Neural Network (CNN) using Keras in Python (and R) 
to classify Fashion MNIST images into 10 categories.


## How to Run (Python)
1. Install dependencies:
   pip install tensorflow keras matplotlib numpy
2. Run the script:
   python fashion_mnist_cnn.py

## How to Run (R)
1. Install keras package in R.
2. Run:
   Rscript fashion_mnist_cnn.R

## Predictions
The model predicts at least two images from the Fashion MNIST test set, showing predicted vs. true labels.

## Files
- `fashion_mnist_cnn.py` → Python code
- `fashion_mnist_cnn.R` → R code
- `README.md` → Instructions

Run the R script:
source("fashion_mnist_cnn.R")
 Model Architecture
The CNN model consists of 6 layers:

Conv2D (32 filters, 3×3 kernel, ReLU activation)

MaxPooling2D (2×2 pool size)

Conv2D (64 filters, 3×3 kernel, ReLU activation)

MaxPooling2D (2×2 pool size)

Conv2D (64 filters, 3×3 kernel, ReLU activation)

Dense Output Layer (Flatten → Dense(64) → Dense(10, softmax))

 Dataset Information
Fashion MNIST dataset contains 70,000 grayscale images of 10 fashion categories:

Label	Class Name
0	T-shirt/top
1	Trouser
2	Pullover
3	Dress
4	Coat
5	Sandal
6	Shirt
7	Sneaker
8	Bag
9	Ankle boot

Training set: 60,000 images

Test set: 10,000 images

Image size: 28×28 pixels

 Expected Performance
After training for ~10 epochs:

Training accuracy: 92–94%

Test accuracy: 90–92%

Training time: 2–5 minutes per epoch (depending on hardware)

 Output Files
Running the code generates:

Log files:

python_output_YYYYMMDD_HHMMSS.txt

r_output_YYYYMMDD_HHMMSS.txt

Prediction visualization:

predictions.png → Shows two test images with predicted vs. true labels

 Custom Class Implementation
Python Class: FashionMNISTClassifier
build_model() → Constructs the 6-layer CNN

train() → Trains the model

evaluate_model() → Evaluates performance

predict_images() → Predicts on new images

get_predictions_details() → Returns prediction details

R Class-like Structure: FashionMNISTModel
Builds a 6-layer CNN

Training method

Evaluation method

Prediction method

Outputs detailed results

 Code Features
Comprehensive Logging → All output saved to timestamped files

Detailed Predictions → Top-3 predictions with confidence scores

Visualization → Side-by-side predicted vs. true labels

Error Handling → Proper exceptions/warning suppression

Modular Design → Easy to extend/modify architecture

