import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("heart_attack_data.csv")  # Ensure the correct file path

# Define features (X) and target (y)
X = df.drop(columns=["heart_attack"])  
y = df["heart_attack"]  # Target variable

# Split data into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Logistic Regression model
clf = LogisticRegression(random_state=42, max_iter=1000)
clf.fit(X_train, y_train)

# Make predictions on test data
y_pred = clf.predict(X_test)

# Evaluate model performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# Function to predict if a new patient is at risk of a heart attack
def predict_heart_attack():
    print("\nEnter patient details:")
    input_data = []
    
    # Define validation rules for each feature
    validation_rules = {
        "age": (1, 130),
        "total_cholesterol": (0, 600),
        "blood_pressure": (0, 300),
        # Add more rules for other features as needed
    }
    
    # Ask user to input values for each feature
    for col in X.columns:
        while True:
            try:
                value = float(input(f"Enter {col}: "))
                if col in validation_rules:
                    min_val, max_val = validation_rules[col]
                    if not (min_val <= value <= max_val):
                        raise ValueError(f"Value for {col} must be between {min_val} and {max_val}.")
                if value < 0:
                    raise ValueError("Value cannot be negative.")
                input_data.append(value)
                break
            except ValueError as e:
                print(f"Invalid input: {e}. Please enter a valid number.")

    # Convert to DataFrame for model prediction
    input_df = pd.DataFrame([input_data], columns=X.columns)
    
    # Predict probability
    prediction_proba = clf.predict_proba(input_df)[0]
    
    # Display result
    high_risk_proba = prediction_proba[1] * 100
    low_risk_proba = prediction_proba[0] * 100
    print(f"\n🔴 High Risk Probability: {high_risk_proba:.2f}%")
    print(f"🟢 Low Risk Probability: {low_risk_proba:.2f}%")

# Call the function to allow user input
predict_heart_attack()