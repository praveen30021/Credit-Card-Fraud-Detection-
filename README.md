# 💳 Credit Card Fraud Detection

This project implements a machine learning pipeline to detect fraudulent credit card transactions using advanced techniques such as XGBoost, Artificial Neural Networks (ANN), and robust preprocessing strategies. The goal is to accurately identify fraud while handling class imbalance and preserving model generalizability.

## 📌 Project Overview

- **Objective:** Detect fraudulent credit card transactions with high accuracy and robustness.
- **Dataset:** Publicly available anonymized credit card dataset (e.g., [Kaggle Credit Card Fraud Dataset](https://www.kaggle.com/mlg-ulb/creditcardfraud)).
- **Tech Stack:** Python, Pandas, Scikit-learn, XGBoost, TensorFlow/Keras, Matplotlib, Seaborn

## 🧠 Key Features

- **Data Preprocessing**
  - One-Hot Encoding for categorical features
  - Robust Scaler normalization
  - SMOTE for handling severe class imbalance
  - Z-test for statistical feature selection

- **Modeling**
  - Trained **XGBoost** and **ANN** classifiers
  - Tuned hyperparameters via GridSearchCV
  - Evaluated using metrics like F1 Score, AUC-ROC, and confusion matrix

- **Performance**
  - Achieved **F1 Score: 0.86**
  - Achieved **AUC Score: 0.92**
  - Improved fraud identification rate by **25%** over the baseline model

## 📊 Results

The optimized pipeline achieved high precision and recall on imbalanced datasets, demonstrating its potential in real-time fraud detection systems.

## 📁 Repository Structure

```
├── FraudDetection.ipynb       # Main notebook with full pipeline
├── README.md                  # Project description and instructions
├── requirements.txt           # Required Python libraries
```

## 🚀 How to Run

1. Clone this repo:
   ```bash
   git clone https://github.com/your-username/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Open and run the notebook:
   ```bash
   jupyter notebook FraudDetection.ipynb
   ```

## 📌 Future Work

- Model explainability using SHAP/LIME
- Deploying as a Flask API
- Real-time detection using streaming data (Kafka/Spark)

## 🙌 Acknowledgments

Thanks to the ULB Machine Learning Group for the dataset and the open-source ML community for tools and libraries.