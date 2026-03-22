"""
Model Evaluation and Visualization
Evaluates trained models and generates insights
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import sys
import warnings
import joblib

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import (confusion_matrix, classification_report, 
                            accuracy_score, f1_score, roc_auc_score, roc_curve)

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def load_data_and_models():
    """Load processed data and trained models"""
    print("Loading data and models...")
    
    df = pd.read_csv('data/grievances_processed.csv')
    
    # Load models
    clf_model = joblib.load('models/classification_random_forest.pkl')
    priority_model = joblib.load('models/priority_random_forest.pkl')
    mappings = joblib.load('models/label_mappings.pkl')
    
    print("✓ Data and models loaded")
    
    return df, clf_model, priority_model, mappings


def prepare_evaluation_data(df):
    """Prepare data for evaluation"""
    y_category = df['category']
    y_priority = df['priority']
    
    # Prepare features
    drop_cols = ['complaint_id', 'category', 'priority', 'location']
    X = df.drop(columns=drop_cols, errors='ignore')
    
    # Encode targets
    category_mapping = {cat: i for i, cat in enumerate(sorted(y_category.unique()))}
    priority_mapping = {'Low': 0, 'Medium': 1, 'High': 2}
    
    y_category_encoded = y_category.map(category_mapping)
    y_priority_encoded = y_priority.map(priority_mapping)
    
    return X, y_category_encoded, y_priority_encoded, category_mapping, priority_mapping


def create_confusion_matrix_plot(y_true, y_pred, labels, output_name):
    """Create confusion matrix visualization"""
    print(f"\n   Creating confusion matrix for {output_name}...")
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    plt.title(f'Confusion Matrix - {output_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    output_path = f'visualizations/confusion_matrix_{output_name.lower().replace(" ", "_")}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_path}")
    plt.close()


def create_classification_report_viz(y_true, y_pred, labels, output_name):
    """Create classification report visualization"""
    print(f"\n   Creating classification report for {output_name}...")
    
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    # Extract metrics
    metrics_data = []
    for label in labels:
        label_str = str(label)
        if label_str in report:
            metrics_data.append({
                'Class': str(label),
                'Precision': report[label_str]['precision'],
                'Recall': report[label_str]['recall'],
                'F1-Score': report[label_str]['f1-score']
            })
        elif label in report:
            metrics_data.append({
                'Class': str(label),
                'Precision': report[label]['precision'],
                'Recall': report[label]['recall'],
                'F1-Score': report[label]['f1-score']
            })
    
    # Create bar plot
    if metrics_data:
        metrics_df = pd.DataFrame(metrics_data)
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(metrics_df))
        width = 0.25
        
        ax.bar(x - width, metrics_df['Precision'], width, label='Precision')
        ax.bar(x, metrics_df['Recall'], width, label='Recall')
        ax.bar(x + width, metrics_df['F1-Score'], width, label='F1-Score')
        
        ax.set_ylabel('Score')
        ax.set_title(f'Classification Metrics - {output_name}')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_df['Class'], rotation=45)
        ax.legend()
        ax.set_ylim(0, 1)
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        output_path = f'visualizations/classification_report_{output_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_path}")
        plt.close()
    else:
        print(f"   ⚠️  No metrics data found for {output_name}")


def analyze_grievance_patterns(df):
    """Analyze and visualize grievance patterns"""
    print("\n📊 Analyzing grievance patterns...")
    
    # Category distribution
    print("   Creating category distribution plot...")
    category_counts = df['category'].value_counts()
    
    fig = px.bar(x=category_counts.index, y=category_counts.values,
                labels={'x': 'Category', 'y': 'Count'},
                title='Grievance Distribution by Category',
                color=category_counts.values, color_continuous_scale='Viridis')
    
    output_path = 'visualizations/category_distribution.html'
    fig.write_html(output_path)
    print(f"   ✓ Saved: {output_path}")
    
    # Priority distribution
    print("   Creating priority distribution plot...")
    priority_counts = df['priority'].value_counts()
    
    colors = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}
    priority_colors = [colors.get(p, 'gray') for p in priority_counts.index]
    
    fig = px.pie(values=priority_counts.values, names=priority_counts.index,
                title='Grievance Distribution by Priority',
                color=priority_counts.index, color_discrete_map=colors)
    
    output_path = 'visualizations/priority_distribution.html'
    fig.write_html(output_path)
    print(f"   ✓ Saved: {output_path}")
    
    # Location distribution (top 10)
    print("   Creating location analysis plot...")
    location_counts = df['location'].value_counts().head(10)
    
    fig = px.bar(y=location_counts.index[::-1], x=location_counts.values[::-1],
                 orientation='h',
                 labels={'x': 'Number of Grievances', 'y': 'Location'},
                 title='Top 10 Locations by Grievance Count')
    
    output_path = 'visualizations/location_distribution.html'
    fig.write_html(output_path)
    print(f"   ✓ Saved: {output_path}")
    
    # Create category-priority heatmap
    print("   Creating category-priority heatmap...")
    category_priority = pd.crosstab(df['category'], df['priority'])
    
    fig = px.imshow(category_priority,
                   labels=dict(x="Priority", y="Category", color="Count"),
                   title="Heatmap: Category vs Priority",
                   color_continuous_scale="YlOrRd")
    
    output_path = 'visualizations/category_priority_heatmap.html'
    fig.write_html(output_path)
    print(f"   ✓ Saved: {output_path}")


def create_feature_importance_plot(model, X, output_name):
    """Create feature importance visualization"""
    print(f"\n   Creating feature importance plot for {output_name}...")
    
    # Get feature importances if available
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X.columns
        
        # Sort
        indices = np.argsort(importances)[-15:]  # Top 15
        
        plt.figure(figsize=(10, 8))
        plt.title(f'Feature Importance - {output_name}')
        plt.barh(range(len(indices)), importances[indices], align='center')
        plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
        plt.xlabel('Importance')
        plt.tight_layout()
        
        output_path = f'visualizations/feature_importance_{output_name.lower().replace(" ", "_")}.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"   ✓ Saved: {output_path}")
        plt.close()


def print_evaluation_summary(y_true, y_pred, task_name):
    """Print evaluation metrics"""
    print(f"\n{'='*50}")
    print(f"{task_name.upper()} - EVALUATION METRICS")
    print(f"{'='*50}")
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))


def main():
    """Run complete evaluation pipeline"""
    print("="*70)
    print("MODEL EVALUATION & VISUALIZATION PIPELINE")
    print("="*70)
    
    try:
        os.makedirs('visualizations', exist_ok=True)
        
        # Load data and models
        df, clf_model, priority_model, mappings = load_data_and_models()
        
        # Prepare data
        X, y_category, y_priority, cat_mapping, pri_mapping = prepare_evaluation_data(df)
        
        # Get predictions
        print("\n🔮 Generating predictions...")
        clf_predictions = clf_model.predict(X)
        priority_predictions = priority_model.predict(X)
        print("✓ Predictions generated")
        
        # Create confusion matrices
        print("\n📊 Creating confusion matrices...")
        category_labels = sorted(df['category'].unique())
        create_confusion_matrix_plot(y_category, clf_predictions, category_labels, "Category Classification")
        create_confusion_matrix_plot(y_priority, priority_predictions, [0, 1, 2], "Priority Prediction")
        
        # Create classification reports
        print("\n📈 Creating classification reports...")
        create_classification_report_viz(y_category, clf_predictions, category_labels, "Category Classification")
        create_classification_report_viz(y_priority, priority_predictions, [0, 1, 2], "Priority Prediction")
        
        # Analyze patterns
        analyze_grievance_patterns(df)
        
        # Feature importance
        print("\n🔍 Creating feature importance plots...")
        create_feature_importance_plot(clf_model, X, "Category Classifier")
        create_feature_importance_plot(priority_model, X, "Priority Predictor")
        
        # Print summaries
        print_evaluation_summary(y_category, clf_predictions, "Category Classification")
        print_evaluation_summary(y_priority, priority_predictions, "Priority Prediction")
        
        print("\n" + "="*70)
        print("✅ EVALUATION COMPLETE")
        print("="*70)
        print(f"\n📁 Visualizations saved to: visualizations/")
        print("\nGenerated files:")
        for file in os.listdir('visualizations')[:10]:
            print(f"   - {file}")
        
    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        raise


if __name__ == "__main__":
    main()
