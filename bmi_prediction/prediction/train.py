import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import LabelEncoder

# Define file paths for model and scaler
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")

# ✅ 1️⃣ Load Dataset
try:
    DATA_FILE_PATH = os.path.join(os.path.dirname(__file__), 'obesity_dataset.csv')

    data = pd.read_csv(DATA_FILE_PATH)

    # Ensure BMI and Outcome are in the dataset
    if "BMI" not in data.columns or "Obesity" not in data.columns:
        raise ValueError("Dataset must contain 'BMI' and 'Outcome' columns")

    X = data[['BMI']].values  # Explicitly select only BMI
    y = data['Obesity'].values  # Ensure correct target variable

    # Encoding the Target Variable (Yes -> 1, No -> 0)
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # ✅ 2️⃣ Split Dataset (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ✅ 3️⃣ Scale Data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ✅ 4️⃣ Train Model (Simple Linear Regression)
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # ✅ 5️⃣ Evaluate Model
    y_pred = model.predict(X_test_scaled)
    model_accuracy = round(r2_score(y_test, y_pred), 4)  # R² Score

    print(f"✅ Model trained successfully! R² Score: {model_accuracy}")
    
    # ✅ 6️⃣ Save Model and Scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

except Exception as e:
    print(f"❌ Error in training: {e}")
    model = None
    scaler = None
    model_accuracy = None
