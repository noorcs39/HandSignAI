# HandSignAI: Hybrid CNN-PSO Framework for American Sign Language Recognition Using Sign Language MNIST Dataset

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A hybrid deep learning approach for classifying American Sign Language (ASL) hand gestures using a Convolutional Neural Network (CNN) optimized with Particle Swarm Optimization (PSO), followed by fine-tuning through backpropagation.

## 🌐 Live Demo / Contact
- 🔗 Website: [https://noorcs39.github.io/Nooruddin](https://noorcs39.github.io/Nooruddin)
- 📬 Contact: Nooruddin – noor.cs2@yahoo.com

---

## 📂 Dataset

- **Source**: [Sign Language MNIST - Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist)
- **Format**: 28x28 grayscale images of hand gestures for ASL alphabets A–Y (excluding J and Z due to motion)
- **File Used**: `sign_mnist_train.csv`

---

## 🧠 Model Architecture

- **Conv2D Layer (32 filters)** → MaxPooling2D
- **Conv2D Layer (64 filters)** → MaxPooling2D
- **Flatten → Dense(128)** → Output Layer (25 classes)
- **Activation**: ReLU + Softmax
- **Optimizer**:
  - Phase 1: Particle Swarm Optimization (PSO)
  - Phase 2: Backpropagation (Adam)

---

## ⚙️ Project Workflow

1. 📥 Load and preprocess data (normalize, reshape, one-hot encode)
2. 🧱 Create a CNN model using Keras
3. 🔀 Optimize initial weights using Particle Swarm Optimization (PSO)
4. 🔧 Fine-tune using backpropagation
5. 📊 Evaluate model performance
6. 📈 Visualize accuracy, F1-score, and confusion matrix

---

## 📊 Evaluation Metrics

- ✅ Accuracy
- ✅ Weighted F1 Score
- ✅ Confusion Matrix Visualization

---

## 🗀️ Sample Inputs

The model was trained using the below ASL gestures from the dataset:

![Sample Gestures](https://github.com/Noorcs39/HandSignAI/assets/sample-asl-image.png)

---

## 🔍 Requirements

```bash
pip install numpy pandas matplotlib scikit-learn tensorflow
```

---

## 🚀 Run the Project

```bash
python main.py
```

Make sure the dataset file `sign_mnist_train.csv` is in the same directory.

---

## 📌 Future Improvements

- Integrate webcam-based real-time prediction
- Extend model to handle dynamic gestures (J, Z)
- Deploy using Streamlit or Flask for a web interface

---

## 📝 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

