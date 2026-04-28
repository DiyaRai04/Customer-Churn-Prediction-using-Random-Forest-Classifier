import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Load dataset
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Preprocessing
def preprocess_data(df):
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Drop non-predictive columns
    df = df.drop(['CustomerID', 'LastPurchaseDate'], axis=1, errors='ignore')
    
    # Handle missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    
    # Identify categorical and numerical columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # Remove target from numerical if present
    if 'Churn' in numerical_cols:
        numerical_cols.remove('Churn')
    
    # Label Encoding for categorical features
    label_encoders = {}
    for col in categorical_cols:
        if col != 'Churn':  # Don't encode target yet
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    
    # Encode target (Churn)
    if df['Churn'].dtype == 'object':
        df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    
    return df, label_encoders, categorical_cols, numerical_cols

# Train Model
def train_model(df, label_encoders):
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    feature_names = X.columns.tolist()
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    
    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'   # Good for imbalanced churn data
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluation
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    print(f"Model Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {auc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save model, scaler, label encoders, and feature names
    joblib.dump(model, 'models/model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(label_encoders, 'models/label_encoders.pkl')
    joblib.dump(feature_names, 'models/feature_names.pkl')
    
    return model, scaler, feature_names

if __name__ == "__main__":
    df = load_data('data/churn prediction.csv')
    df, encoders, cat_cols, num_cols = preprocess_data(df)
    model, scaler, feature_names = train_model(df, encoders)
    print("✅ Model training completed and saved!")