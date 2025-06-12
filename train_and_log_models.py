import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.keras
import joblib
import os
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load and prepare data
data = pd.read_csv("hand_landmarks_data.csv")
X = data.drop("label", axis=1)
y = data["label"]

# Encode labels
encoder = LabelEncoder()
y = encoder.fit_transform(y)
joblib.dump(encoder, "label_encoder.pkl")

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define MLflow experiment
mlflow.set_experiment("Hand Gesture Recognition")

def log_model(model, name, params=None):
    with mlflow.start_run(run_name=name):
        if params:
            mlflow.log_params(params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        mlflow.sklearn.log_model(model, name)
        print(f"{name} -> Accuracy: {acc:.4f}, F1: {f1:.4f}")

# Train & log SVM
log_model(SVC(probability=True), "SVM")

# Train & log Random Forest
log_model(RandomForestClassifier(n_estimators=100, random_state=42), "RandomForest")

# Train & log XGBoost
log_model(xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'), "XGBoost")

# Train & log Keras FFNN
with mlflow.start_run(run_name="Keras_FFNN"):
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(64, activation='relu'),
        Dense(len(np.unique(y)), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1, verbose=0)

    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test), axis=1)
    f1 = f1_score(y_test, y_pred, average="weighted")

    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)
    mlflow.keras.log_model(model, "Keras_FFNN")

    print(f"Keras_FFNN -> Accuracy: {acc:.4f}, F1: {f1:.4f}")
    model.save("keras_ffnn_model.keras")
