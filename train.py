import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    mean_absolute_error, mean_squared_error, r2_score
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRegressor

# Directories and paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)
DATA_FILE = os.path.join(DATA_DIR, 'seattle-weather.csv')
MODEL_BUNDLE_PATH = os.path.join(BASE_DIR, 'models_bundle.pkl')
METRICS_PATH = os.path.join(BASE_DIR, 'model_metrics.json')

def load_data():
    """Load the official stable Seattle weather dataset directly from Vega repository."""
    url = 'https://raw.githubusercontent.com/vega/vega-datasets/next/data/seattle-weather.csv'
    print(f"Downloading stable Seattle rainfall dataset from:\n{url}")
    try:
        df = pd.read_csv(url)
        df.to_csv(DATA_FILE, index=False)
        print(f"Dataset successfully downloaded and cached to: {DATA_FILE}")
    except Exception as e:
        print(f"Network download failed: {e}. Trying local cache...")
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            print(f"Loaded dataset from local cache: {DATA_FILE}")
        else:
            raise e

    print(f"Dataset Shape: {df.shape}")
    print("Columns:", list(df.columns))
    return df

def train_and_evaluate():
    df = load_data()

    # 1. Feature Engineering: Extract date parts exactly as specified
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    # 2. Define Features (X) - excluding original date and targets
    feature_cols = ['temp_max', 'temp_min', 'wind', 'year', 'month', 'day']
    X = df[feature_cols]

    # 3. Define Targets (y)
    # Target A: Regression (predict continuous precipitation amount)
    y_regression = df['precipitation']

    # Target B: Classification (predict whether it rains - 1 for rain/drizzle, 0 otherwise)
    y_classification = df['weather'].apply(lambda x: 1 if x in ['rain', 'drizzle'] else 0)

    # 4. Train-Test Splits
    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X, y_regression, test_size=0.2, random_state=42
    )

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X, y_classification, test_size=0.2, random_state=42, stratify=y_classification
    )

    print("\n--- Regression Split ---")
    print(f"Train: {X_train_reg.shape}, Test: {X_test_reg.shape}")
    print("--- Classification Split ---")
    print(f"Train: {X_train_clf.shape}, Test: {X_test_clf.shape}")

    # 5. Define and Train Classification Models
    print("\n=== Training Classification Models ===")
    clf_models = {
        'random_forest': RandomForestClassifier(random_state=42, n_estimators=100),
        'xgboost': XGBClassifier(random_state=42, n_estimators=100, eval_metric='logloss'),
        'logistic_regression': LogisticRegression(random_state=42, max_iter=1000),
        'decision_tree': DecisionTreeClassifier(random_state=42)
    }

    clf_metrics = {}
    for name, model in clf_models.items():
        model.fit(X_train_clf, y_train_clf)
        y_pred = model.predict(X_test_clf)
        acc = accuracy_score(y_test_clf, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(y_test_clf, y_pred, average='binary')
        clf_metrics[name] = {
            'accuracy': round(float(acc) * 100, 2),
            'precision': round(float(prec) * 100, 2),
            'recall': round(float(rec) * 100, 2),
            'f1_score': round(float(f1) * 100, 2)
        }
        print(f"[{name.upper()}] Accuracy: {acc:.2%}, F1: {f1:.2%}")

    # 6. Define and Train Regression Models
    print("\n=== Training Regression Models ===")
    reg_models = {
        'random_forest': RandomForestRegressor(random_state=42, n_estimators=100),
        'xgboost': XGBRegressor(random_state=42, n_estimators=100),
        'linear_regression': LinearRegression(),
        'ridge': Ridge(alpha=1.0, random_state=42)
    }

    reg_metrics = {}
    for name, model in reg_models.items():
        model.fit(X_train_reg, y_train_reg)
        y_pred = model.predict(X_test_reg)
        y_pred_clipped = np.clip(y_pred, 0, None)
        mae = mean_absolute_error(y_test_reg, y_pred_clipped)
        rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_clipped))
        r2 = r2_score(y_test_reg, y_pred_clipped)
        reg_metrics[name] = {
            'mae': round(float(mae), 2),
            'rmse': round(float(rmse), 2),
            'r2_score': round(float(r2), 3)
        }
        print(f"[{name.upper()}] MAE: {mae:.2f} mm, RMSE: {rmse:.2f} mm, R2: {r2:.3f}")

    # 7. Save model bundle & metrics
    bundle = {
        'feature_cols': feature_cols,
        'clf_models': clf_models,
        'reg_models': reg_models,
        'clf_metrics': clf_metrics,
        'reg_metrics': reg_metrics
    }
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"\nSaved models bundle to: {MODEL_BUNDLE_PATH}")

    with open(METRICS_PATH, 'w') as f:
        json.dump({'classification': clf_metrics, 'regression': reg_metrics}, f, indent=2)
    print(f"Saved metrics summary to: {METRICS_PATH}")

if __name__ == '__main__':
    train_and_evaluate()
