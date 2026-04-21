# Breast Cancer Detection using Neural Network

A deep learning project that classifies breast tumors as **Malignant** or **Benign** using a fully connected neural network trained on the Wisconsin Breast Cancer dataset.

---

## Overview

This project builds and trains a neural network using TensorFlow/Keras to detect breast cancer from 30 numerical features extracted from digitized images of fine needle aspirates (FNA) of breast masses. The trained model is saved and used in a predictive system that takes input data and outputs a diagnosis.

---

## Dataset

- **Source:** `sklearn.datasets.load_breast_cancer()`
- **Samples:** 569
- **Features:** 30 (mean, standard error, and worst values of cell nucleus measurements)
- **Target Classes:**
  - `0` → Malignant
  - `1` → Benign

---

## Project Structure

```
breast-cancer-detection/
│
├── breast_cancer_detection_using_nn.ipynb   # Main notebook
├── breast_cancer_model.keras                # Saved trained model
├── requirements.txt                         # Python dependencies
└── README.md
```

---

## Model Architecture

```
Input (30 features)
    ↓
Flatten Layer
    ↓
Dense Layer (20 units, ReLU activation)
    ↓
Dense Layer (2 units, Sigmoid activation)
    ↓
Output: Malignant / Benign
```

- **Optimizer:** Adam
- **Loss Function:** Sparse Categorical Crossentropy
- **Epochs:** 10
- **Validation Split:** 10%

---

## Workflow

1. **Data Collection** — Load breast cancer dataset from sklearn
2. **Preprocessing** — Handle missing values, explore distributions, standardize features using `StandardScaler`
3. **Train/Test Split** — 80% training / 20% testing
4. **Model Training** — Train the neural network and track accuracy & loss
5. **Evaluation** — Evaluate on test data and visualize training curves
6. **Prediction** — Classify new input samples as Malignant or Benign
7. **Model Saving** — Save model as `.keras` file for reuse

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/breast-cancer-detection.git
   cd breast-cancer-detection
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebook**
   ```bash
   jupyter notebook breast_cancer_detection_using_nn.ipynb
   ```

4. **Or load the saved model directly**
   ```python
   import tensorflow as tf
   model = tf.keras.models.load_model('breast_cancer_model.keras')
   ```

---

## Results

The model is trained for 10 epochs and evaluated on the held-out test set. Training and validation accuracy/loss curves are plotted to monitor model performance.

---

## Requirements

See [`requirements.txt`](requirements.txt)

---

## License

This project is open-source and available under the [MIT License](LICENSE).
