"""
Model Training & Persistence - Save ML models for deployment
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, accuracy_score, f1_score
import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))
from config import settings


def train_regression_model(data: pd.DataFrame):
    """Train and save regression model for xG prediction"""
    print("\n" + "="*60)
    print("TRAINING REGRESSION MODEL (xG per 90)")
    print("="*60)
    
    # Prepare features
    target_col = 'xg_per90'
    feature_cols = ['goals_per90', 'shots_per90', 'assists_per90', 
                   'progressive_passes_per90', 'carries_per90']
    
    # Filter available columns
    available_features = [col for col in feature_cols if col in data.columns]
    
    if not available_features or target_col not in data.columns:
        print("❌ Missing required columns for regression")
        return None
    
    # Prepare data
    X = data[available_features].fillna(0)
    y = data[target_col]
    
    # Remove outliers
    mask = (y >= y.quantile(0.01)) & (y <= y.quantile(0.99))
    X, y = X[mask], y[mask]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train
    print(f"\n📊 Training on {len(X_train)} samples...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"\n✓ Model trained!")
    print(f"  - R² Score: {r2:.3f}")
    print(f"  - MAE: {mae:.3f}")
    
    # Save model
    model_path = settings.MODELS_DIR / "xg_regression_model.pkl"
    joblib.dump({
        'model': model,
        'features': available_features,
        'metrics': {'r2': r2, 'mae': mae}
    }, model_path)
    
    print(f"\n💾 Model saved: {model_path}")
    
    return model


def train_classification_model(data: pd.DataFrame):
    """Train and save classification model for performance tiers"""
    print("\n" + "="*60)
    print("TRAINING CLASSIFICATION MODEL (Performance Tiers)")
    print("="*60)
    
    # Create target (performance tiers based on offensive_score)
    if 'offensive_score' not in data.columns:
        print("❌ Missing offensive_score column")
        return None
    
    # Define tiers
    data = data.copy()
    data['tier'] = pd.cut(
        data['offensive_score'],
        bins=[0, 40, 60, 80, 100],
        labels=['Low', 'Medium', 'High', 'Elite']
    )
    
    # Features
    feature_cols = ['goals_per90', 'assists_per90', 'xg_per90', 
                   'shots_per90', 'progressive_passes_per90']
    
    available_features = [col for col in feature_cols if col in data.columns]
    
    if not available_features:
        print("❌ Missing required features")
        return None
    
    # Prepare
    X = data[available_features].fillna(0)
    y = data['tier']
    
    # Remove NaN targets
    mask = ~y.isna()
    X, y = X[mask], y[mask]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train
    print(f"\n📊 Training on {len(X_train)} samples...")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"\n✓ Model trained!")
    print(f"  - Accuracy: {accuracy:.3f}")
    print(f"  - F1 Score: {f1:.3f}")
    
    # Save
    model_path = settings.MODELS_DIR / "tier_classification_model.pkl"
    joblib.dump({
        'model': model,
        'features': available_features,
        'classes': model.classes_.tolist(),
        'metrics': {'accuracy': accuracy, 'f1': f1}
    }, model_path)
    
    print(f"\n💾 Model saved: {model_path}")
    
    return model


def main():
    """Main training pipeline"""
    print("="*60)
    print("🤖 ML Model Training & Persistence")
    print("="*60)
    
    # Load data
    print("\n📂 Loading data...")
    data_path = settings.PROCESSED_DATA_DIR / "players_season_stats.csv"
    
    if not data_path.exists():
        print(f"❌ Data not found: {data_path}")
        print("   Run ETL pipeline first: python src/etl/etl_pipeline.py")
        return
    
    data = pd.read_csv(data_path)
    print(f"✓ Loaded {len(data)} player records")
    
    # Train models
    reg_model = train_regression_model(data)
    clf_model = train_classification_model(data)
    
    # Summary
    print("\n" + "="*60)
    print("✅ TRAINING COMPLETE!")
    print("="*60)
    
    models_saved = []
    if reg_model is not None:
        models_saved.append("xg_regression_model.pkl")
    if clf_model is not None:
        models_saved.append("tier_classification_model.pkl")
    
    if models_saved:
        print(f"\n💾 Models saved to: {settings.MODELS_DIR}")
        for model_name in models_saved:
            print(f"  - {model_name}")
    
    print("\n🎯 Models ready for deployment!")


if __name__ == "__main__":
    main()
