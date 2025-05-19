# Heart Attack Risk Prediction System

![Healthcare AI](https://img.shields.io/badge/Healthcare-AI-brightgreen) ![Python](https://img.shields.io/badge/Python-3.6%2B-blue) ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-orange)

A machine learning system that predicts heart attack risk using logistic regression.

## Table of Contents
- [Heart Attack Risk Prediction System](#heart-attack-risk-prediction-system)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Data Requirements](#data-requirements)
  - [Output Interpretation](#output-interpretation)
  - [Project Structure](#project-structure)
  - [License](#license)

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

## Usage

1. Place your dataset as `heart_attack_data.csv` in the project root
2. Run the prediction system:
```bash
python index.py
```
3. Follow the prompts to enter patient data

Example output:
```
Model Accuracy: 0.87
Enter patient details:
Enter age: 45
Enter total_cholesterol: 200
...

🔴 High Risk Probability: 72.34%
🟢 Low Risk Probability: 27.66%
```

## Data Requirements
- CSV file named `heart_attack_data.csv`
- Must contain:
  - Features: Various health metrics (numerical values)
  - Target column named "heart_attack" (binary: 1=risk, 0=no risk)

## Output Interpretation
| Output | Meaning |
|--------|---------|
| 🔴 High Risk Probability | Chance of heart attack occurring |
| 🟢 Low Risk Probability | Chance of no heart attack |

## Project Structure
```
.
├── index.py    # Main prediction script
├── heart_attack_data.csv        # Training dataset
├── README.md                   # This documentation
```

## License
This project is licensed under the MIT License
