"""
Machine Learning Classification Models Training Script
Dataset: Wine Quality Dataset from UCI Repository
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score, 
                              recall_score, f1_score, matthews_corrcoef)
import pickle
import warnings
warnings.filterwarnings('ignore')

def load_and_prepare_data():
    """Load and prepare the Wine Quality dataset"""
    # URLs for Wine Quality dataset (red and white wine)
    url_red = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    url_white = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv"
    
    # Load both datasets
    df_red = pd.read_csv(url_red, sep=';')
    df_white = pd.read_csv(url_white, sep=';')
    
    # Add wine type feature
    df_red['wine_type'] = 1  # Red wine
    df_white['wine_type'] = 0  # White wine
    
    # Combine datasets
    df = pd.concat([df_red, df_white], ignore_index=True)
    
    print(f"Original dataset shape: {df.shape}")
    print(f"Red wine samples: {len(df_red)}")
    print(f"White wine samples: {len(df_white)}")
    print(f"Total samples: {len(df)}")
    
    # Target variable: 'quality' (score between 3-9)
    # Convert to binary classification: Good (>=6) vs Poor (<6)
    df['target'] = (df['quality'] >= 6).astype(int)
    df = df.drop('quality', axis=1)
    
    print(f"\nAfter preprocessing: {df.shape}")
    print(f"Number of instances: {df.shape[0]}")
    print(f"Number of features: {df.shape[1] - 1}")
    
    # Feature names
    feature_names = [col for col in df.columns if col != 'target']
    print(f"\nFeatures ({len(feature_names)}): {feature_names}")
    
    return df

def evaluate_model(y_true, y_pred, y_pred_proba):
    """Calculate all evaluation metrics"""
    metrics = {
        'Accuracy': accuracy_score(y_true, y_pred),
        'AUC': roc_auc_score(y_true, y_pred_proba),
        'Precision': precision_score(y_true, y_pred, zero_division=0),
        'Recall': recall_score(y_true, y_pred, zero_division=0),
        'F1': f1_score(y_true, y_pred, zero_division=0),
        'MCC': matthews_corrcoef(y_true, y_pred)
    }
    return metrics

def train_all_models():
    """Train all 6 classification models and save results"""
    
    # Load data
    print("Loading and preparing Wine Quality dataset...")
    df = load_and_prepare_data()
    
    print(f"\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    print(f"Dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")
    print(f"Number of instances: {df.shape[0]}")
    print(f"\nClass distribution:\n{df['target'].value_counts()}")
    print(f"\nClass names: 0=Poor Quality (<6), 1=Good Quality (>=6)")
    print(f"Class balance:\n{df['target'].value_counts(normalize=True).round(3)}")
    
    # Prepare features and target
    X = df.drop('target', axis=1)
    y = df['target']
    
    # Split data (using 80-20 split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    
    # Dictionary to store all models and results
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'K-Nearest Neighbor': KNeighborsClassifier(n_neighbors=5),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, eval_metric='logloss')
    }
    
    results = {}
    
    print("\nTraining models and calculating metrics...\n")
    
    for name, model in models.items():
        print(f"Training {name}...")
        
        # Train model
        if name in ['Logistic Regression', 'K-Nearest Neighbor']:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
        else:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        metrics = evaluate_model(y_test, y_pred, y_pred_proba)
        results[name] = metrics
        
        # Save model
        model_filename = f"model/{name.lower().replace(' ', '_').replace('-', '_')}_model.pkl"
        with open(model_filename, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"  Accuracy: {metrics['Accuracy']:.4f}")
        print(f"  AUC: {metrics['AUC']:.4f}")
        print(f"  Precision: {metrics['Precision']:.4f}")
        print(f"  Recall: {metrics['Recall']:.4f}")
        print(f"  F1: {metrics['F1']:.4f}")
        print(f"  MCC: {metrics['MCC']:.4f}\n")
    
    # Save results to CSV
    results_df = pd.DataFrame(results).T
    results_df.to_csv('model/model_results.csv')
    
    # Save test data for app
    test_data = {
        'X_test': X_test,
        'y_test': y_test,
        'X_test_scaled': X_test_scaled,
        'feature_names': X.columns.tolist()
    }
    with open('model/test_data.pkl', 'wb') as f:
        pickle.dump(test_data, f)
    
    print("\nAll models trained and saved successfully!")
    print(f"\nResults saved to: model/model_results.csv")
    
    return results_df

if __name__ == "__main__":
    results_df = train_all_models()
    print("\n" + "="*80)
    print("FINAL RESULTS - MODEL COMPARISON")
    print("="*80)
    print(results_df.round(4))
