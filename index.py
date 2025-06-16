import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_curve, auc, confusion_matrix

# === 1. Load Dataset ===
df = pd.read_csv("heart_attack_data.csv")  # Update with your file path

# === 2. Data Visualization ===
print("\n📊 Dataset Overview:")
print(df.head())
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# Distribution of target
sns.countplot(data=df, x="heart_attack")
plt.title("Target Class Distribution")
plt.show()

# === 3. Feature/Target Split ===
X = df.drop(columns=["heart_attack"])
y = df["heart_attack"]

# === 4. Train/Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 5. Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === 6. Model Training ===
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train_scaled, y_train)

# === 7. Evaluation ===
y_pred = clf.predict(X_test_scaled)
y_prob = clf.predict_proba(X_test_scaled)[:, 1]

print(f"\n Accuracy: {accuracy_score(y_test, y_pred):.2f}")
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "k--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.show()

# Coefficients
coef_df = pd.DataFrame({"Feature": X.columns, "Coefficient": clf.coef_[0]})
coef_df["Abs_Coeff"] = np.abs(coef_df["Coefficient"])
print("\n🔍 Top Contributing Features:")
print(coef_df.sort_values("Abs_Coeff", ascending=False).head(10))

# === 8. Save model and scaler ===
joblib.dump(clf, "heart_attack_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(X.columns.tolist(), "features.pkl")
print("\n Model and scaler saved to disk.")

# === 9. Prediction Interface ===
def predict_heart_attack():
    print("\n Enter patient data to assess heart attack risk:")
    input_data = []

    validation_rules = {
        "age": (1, 130),
        "total_cholesterol": (0, 600),
        "blood_pressure": (0, 300),
        # Add more if needed
    }

    for col in X.columns:
        while True:
            try:
                val = float(input(f"Enter {col}: "))
                if col in validation_rules:
                    min_v, max_v = validation_rules[col]
                    if not (min_v <= val <= max_v):
                        raise ValueError(f"Value must be between {min_v} and {max_v}.")
                input_data.append(val)
                break
            except ValueError as e:
                print(f" {e}")

    input_df = pd.DataFrame([input_data], columns=X.columns)
    input_scaled = scaler.transform(input_df)
    prob = clf.predict_proba(input_scaled)[0]

    print(f"\n Heart Attack Risk: {prob[1]*100:.2f}%")
    print("✅ Safe" if prob[1] < 0.5 else "⚠️ High Risk!")

if __name__ == "__main__":
    predict_heart_attack()
