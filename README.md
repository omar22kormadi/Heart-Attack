# Heart Attack Risk Prediction System

![Healthcare AI](https://img.shields.io/badge/Healthcare-AI-brightgreen) ![Python](https://img.shields.io/badge/Python-3.6%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)

A machine learning system that predicts heart attack risk using logistic regression.

## Table of Contents
- [Heart Attack Risk Prediction System](#heart-attack-risk-prediction-system)
  - [Table of Contents](#table-of-contents)
  - [Files](#files)
  - [Features](#features)
  - [Installation](#installation)
  - [How to Run](#how-to-run)
  - [Example Output](#example-output)
  - [Data Requirements](#data-requirements)
  - [Author](#author)

---

## Files

| File | Description |
|------|-------------|
| `index.py` | Main code (training + evaluation + prediction) |
| `heart_attack_data.csv` | Input dataset (must be added manually) |
| `heart_attack_model.pkl` | Saved trained model |
| `scaler.pkl` | Saved StandardScaler used during training |
| `features.pkl` | List of feature names used by the model |

---

## Features
✔️ Logistic Regression model trained on medical data  
✔️ Interactive CLI for patient risk assessment  
✔️ Input validation for health metrics  
✔️ Probability-based risk prediction (high/low)  
✔️ Model evaluation metrics  

## Installation

1. Clone the repository:
```bash
git clone https://github.com/omar22kormadi/Heart-Attack.git
```

---

## How to Run

1. Ensure you have Python 3.8+ installed
2. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

3. Put your dataset in the same folder as `index.py` and name it `heart_attack_data.csv`
4. Run the script:

```bash
python index.py
```

---

## Example Output

```bash
✅ Accuracy: 0.87

📋 Classification Report:
              precision    recall  f1-score   support
           0       0.88      0.91      0.89       105
           1       0.85      0.81      0.83        75

🔍 Top Contributing Features:
       Feature  Coefficient
3        age         1.456
5  cholesterol        1.123
7         bp          0.983
...
```

---

## Data Requirements
- CSV file named `heart_attack_data.csv`
- Must contain:
  - Features: Various health metrics (numerical values)
  - Target column named "heart_attack" (binary: 1=risk, 0=no risk)




##  Author

Developed by Amor Kormadi (2025) 
