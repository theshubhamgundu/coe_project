"""
Model Training Pipeline
Trains multiple ML models for classification and priority prediction
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings
import joblib
from datetime import datetime

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_processed_data(input_path='data/grievances_processed.csv'):
    """Load preprocessed data"""
    print(f"Loading processed data from {input_path}...")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Processed data not found. Run preprocessing first.")
    
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} records with {df.shape[1]} features")
    
    return df


def prepare_data(df):
    """Prepare features and target variable"""
    print("\n📊 Preparing training data...")
    
    # Target variables
    y_category = df['category'].copy()
    y_priority = df['priority'].copy()
    
    # Features (drop non-feature columns)
    drop_cols = ['complaint_id', 'category', 'priority', 'location']
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # Encode target variables
    category_mapping = {cat: i for i, cat in enumerate(sorted(y_category.unique()))}
    priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    
    y_category_encoded = y_category.map(category_mapping)
    y_priority_encoded = y_priority.map(priority_mapping)
    
    print(f"✓ Features: {X.shape}")
    print(f"✓ Categories: {len(category_mapping)}")
    print(f"✓ Priority levels: {len(priority_mapping)}")
    
    return X, y_category_encoded, y_priority_encoded, category_mapping, priority_mapping


def train_classification_models(X_train, X_test, y_train, y_test):
    """Train multiple classification models"""
    print("\n🚀 Training classification models...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Naive Bayes': GaussianNB(),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"      Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    
    return trained_models, results


def train_priority_prediction_models(X_train, X_test, y_train, y_test):
    """Train models for priority prediction"""
    print("\n🎯 Training priority prediction models...")
    
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42, verbosity=0)
    }
    
    results = {}
    trained_models = {}
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        
        # Train model
        model.fit(X_train, y_train)
        trained_models[name] = model
        
        # Predictions
        y_pred = model.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        results[name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print(f"      Accuracy: {accuracy:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f}")
    
    return trained_models, results


def save_models(models, task_name='classification'):
    """Save trained models"""
    print(f"\n💾 Saving {task_name} models...")
    
    os.makedirs('models', exist_ok=True)
    
    for name, model in models.items():
        filename = f"models/{task_name}_{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)
        print(f"   ✓ Saved: {filename}")


def print_summary(clf_results, priority_results):
    """Print training summary"""
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    
    print("\n📂 CLASSIFICATION MODELS:")
    print("-" * 70)
    clf_df = pd.DataFrame(clf_results).T
    print(clf_df.to_string())
    best_clf = clf_df['f1_score'].idxmax()
    print(f"\n   🏆 Best Model: {best_clf} (F1-Score: {clf_df.loc[best_clf, 'f1_score']:.4f})")
    
    print("\n⚠️ PRIORITY PREDICTION MODELS:")
    print("-" * 70)
    priority_df = pd.DataFrame(priority_results).T
    print(priority_df.to_string())
    best_priority = priority_df['f1_score'].idxmax()
    print(f"\n   🏆 Best Model: {best_priority} (F1-Score: {priority_df.loc[best_priority, 'f1_score']:.4f})")


def main():
    """Run complete training pipeline"""
    print("="*70)
    print("ML MODEL TRAINING PIPELINE")
    print("="*70)
    
    try:
        # Load data
        df = load_processed_data('data/grievances_processed.csv')
        
        # Prepare data
        X, y_category, y_priority, cat_mapping, pri_mapping = prepare_data(df)
        
        # Split data for category classification
        print("\n✂️  Splitting data (80-20 train-test)...")
        X_train, X_test, y_cat_train, y_cat_test = train_test_split(
            X, y_category, test_size=0.2, random_state=42, stratify=y_category
        )
        
        # Get priority test set
        _, _, _, y_pri_train, y_pri_test = train_test_split(
            X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
        )
        
        print(f"   Train set: {len(X_train)} samples")
        print(f"   Test set: {len(X_test)} samples")
        
        # Train classification models
        clf_models, clf_results = train_classification_models(
            X_train, X_test, y_cat_train, y_cat_test
        )
        
        # Train priority prediction models
        priority_models, priority_results = train_priority_prediction_models(
            X_train, X_test, y_pri_train, y_pri_test
        )
        
        # Save models
        save_models(clf_models, 'classification')
        save_models(priority_models, 'priority')
        
        # Print summary
        print_summary(clf_results, priority_results)
        
        # Save mappings
        print("\n📝 Saving feature mappings...")
        joblib.dump({'category': cat_mapping, 'priority': pri_mapping}, 
                   'models/label_mappings.pkl')
        print("   ✓ Mappings saved")
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
