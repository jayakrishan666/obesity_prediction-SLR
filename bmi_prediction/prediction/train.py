import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# Define CSV file path
DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'obesity_dataset.csv')

def load_and_train_model():
    """Loads dataset, trains the model, and returns the trained model and scaler."""
    
    # Check if dataset exists
    if not os.path.exists(DATA_FILE_PATH):
        print("⚠️ CSV file not found!")
        return None, None, "N/A"

    try:
        data = pd.read_csv(DATA_FILE_PATH)
        print("✅ CSV file loaded successfully.")

        # Ensure required columns exist
        if 'BMI' not in data.columns or 'Obesity' not in data.columns:
            print("⚠️ Missing required columns in dataset!")
            return None, None, "N/A"

        # Convert 'BMI' to numeric
        data['BMI'] = pd.to_numeric(data['BMI'], errors='coerce')
        
        # Convert 'Obesity' to binary (Yes=1, No=0)
        data['Obesity'] = data['Obesity'].map({'Yes': 1, 'No': 0})

        # Handle missing values
        data.dropna(inplace=True)

        # Check if dataset is empty
        if data.empty:
            print("⚠️ Dataset is empty after cleaning!")
            return None, None, "N/A"

        print("✅ Dataset processed successfully.")

        # Prepare features and labels
        X = data[['BMI']]
        y = data['Obesity']

        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Apply feature scaling (Standardization)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train Linear Regression Model
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)

        # Predict on test data
        y_pred = model.predict(X_test_scaled)

        # Compute R² Score (Model Accuracy)
        model_accuracy = round(r2_score(y_test, y_pred), 4)
        
        print(f"✅ Model trained successfully! R² Score: {model_accuracy}")
        return model, scaler, model_accuracy

    except Exception as e:
        print(f"🚨 Error during model training: {e}")
        return None, None, "N/A"

# Train the model when the server starts
model, scaler, model_accuracy = load_and_train_model()

# Debugging Output
if model is None or scaler is None:
    print("🚨 Model or Scaler was not properly trained!")
else:
    print("✅ Model and Scaler are ready to use.")
