"""
Model Prediction Engine
Provides easy interface for making predictions on new grievances
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import TextProcessor, FeatureEngineer


class GrievancePredictionEngine:
    """Predict category and priority for new grievances"""
    
    def __init__(self, model_dir='models'):
        """Load trained models and mappings"""
        self.model_dir = model_dir
        
        # Load models
        self.clf_model = joblib.load(f'{model_dir}/classification_random_forest.pkl')
        self.priority_model = joblib.load(f'{model_dir}/priority_random_forest.pkl')
        self.mappings = joblib.load(f'{model_dir}/label_mappings.pkl')
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.feature_engineer = FeatureEngineer()
        
        # Reverse mappings for decoding
        self.category_reverse_map = {v: k for k, v in self.mappings['category'].items()}
        self.priority_reverse_map = {0: 'Low', 1: 'Medium', 2: 'High'}
        
        # Load TF-IDF vectorizer
        try:
            self.tfidf_vectorizer = joblib.load(f'{model_dir}/tfidf_vectorizer.pkl')
        except:
            self.tfidf_vectorizer = None
    
    def predict(self, complaint_text, location='Unknown', department='Municipal Services'):
        """
        Predict category and priority for a grievance
        """
        
        # Process text
        cleaned_text = self.text_processor.clean_text(complaint_text)
        sentiment = self.text_processor.get_sentiment(complaint_text)
        
        # 1. Text statistics (7 features)
        features = {
            'text_length': len(complaint_text),
            'word_count': len(complaint_text.split()),
            'avg_word_length': len(complaint_text) / len(complaint_text.split()) if len(complaint_text.split()) > 0 else 0,
            'unique_words_ratio': len(set(complaint_text.split())) / len(complaint_text.split()) if len(complaint_text.split()) > 0 else 0,
            'uppercase_ratio': sum(1 for c in complaint_text if c.isupper()) / len(complaint_text) if len(complaint_text) > 0 else 0,
            'digit_count': sum(1 for c in complaint_text if c.isdigit()),
            'special_char_count': sum(1 for c in complaint_text if c in '!?.,;:-'),
        }
        
        # 2. Add Time features (6 features)
        now = datetime.now()
        features.update({
            'hour': now.hour,
            'day_of_week': now.weekday(),
            'day_of_month': now.day,
            'month': now.month,
            'quarter': (now.month - 1) // 3 + 1,
            'is_weekend': 1 if now.weekday() >= 5 else 0
        })
        
        # 3. Add Location features (3 features)
        features.update({
            'location_complaint_count': 0,
            'is_top_location': 0,
            'location_density': 0
        })
        
        # 4. Add Sentiment features (3 features)
        features.update({
            'polarity': sentiment['polarity'],
            'subjectivity': sentiment['subjectivity'],
            'sentiment_intensity': abs(sentiment['polarity'])
        })
        
        # 5. Extract TF-IDF features (100 features)
        tfidf_features = {}
        if self.tfidf_vectorizer:
            tfidf_matrix = self.tfidf_vectorizer.transform([cleaned_text])
            tfidf_values = tfidf_matrix.toarray()[0]
            for i, val in enumerate(tfidf_values):
                tfidf_features[f'tfidf_{i}'] = val
        else:
            for i in range(100):
                tfidf_features[f'tfidf_{i}'] = 0
                
        # 6. Add Department dummies (3 features)
        dept_cols = ['department_Police', 'department_Public Works', 'department_Utilities']
        for col in dept_cols:
            features[col] = 1 if department in col else 0
            
        # 7. Add Sentiment dummies (2 features)
        features['sentiment_neutral'] = 1 if -0.1 <= sentiment['polarity'] <= 0.1 else 0
        features['sentiment_positive'] = 1 if sentiment['polarity'] > 0.1 else 0
        
        # Final combined features (124 features total)
        feature_order = [
            'text_length', 'word_count', 'avg_word_length', 'unique_words_ratio',
            'uppercase_ratio', 'digit_count', 'special_char_count', 'hour',
            'day_of_week', 'day_of_month', 'month', 'quarter', 'is_weekend',
            'location_complaint_count', 'is_top_location', 'location_density',
            'polarity', 'subjectivity', 'sentiment_intensity'
        ]
        
        # Append TF-IDF 100 features
        tfidf_order = [f'tfidf_{i}' for i in range(100)]
        
        # Append dummies
        extra_order = [
            'department_Police', 'department_Public Works', 'department_Utilities',
            'sentiment_neutral', 'sentiment_positive'
        ]
        
        feature_vector = []
        for col in feature_order: feature_vector.append(features[col])
        for col in tfidf_order: feature_vector.append(tfidf_features[col])
        for col in extra_order: feature_vector.append(features[col])
        
        # Make predictions
        category_pred = self.clf_model.predict([feature_vector])[0]
        priority_pred = self.priority_model.predict([feature_vector])[0]
        
        # Get confidence scores
        category_proba = self.clf_model.predict_proba([feature_vector])[0]
        priority_proba = self.priority_model.predict_proba([feature_vector])[0]
        
        return {
            'complaint_text': complaint_text,
            'predicted_category': self.category_reverse_map.get(category_pred, 'Unknown'),
            'category_confidence': float(category_proba[category_pred]),
            'priority_level': self.priority_reverse_map[priority_pred],
            'priority_confidence': float(priority_proba[priority_pred]),
            'sentiment_score': sentiment['polarity'],
            'features': features
        }
    
    def predict_batch(self, complaints_list):
        """Predict for multiple complaints"""
        results = []
        for complaint in complaints_list:
            results.append(self.predict(complaint))
        return results


if __name__ == "__main__":
    # Example usage
    engine = GrievancePredictionEngine()
    
    test_complaints = [
        "Deep pothole on Main Road causing accidents and vehicle damage",
        "Water supply has not been working for 5 days affecting households",
        "Minor parking issue in residential area"
    ]
    
    print("="*70)
    print("GRIEVANCE PREDICTION ENGINE - TEST RUN")
    print("="*70)
    
    for i, complaint in enumerate(test_complaints, 1):
        print(f"\n{'─'*70}")
        print(f"Complaint {i}:")
        print(f"Text: {complaint}")
        
        result = engine.predict(complaint)
        
        print(f"\n📊 Prediction Results:")
        print(f"  Category: {result['predicted_category']} (Confidence: {result['category_confidence']:.2%})")
        print(f"  Priority: {result['priority_level']} (Confidence: {result['priority_confidence']:.2%})")
        print(f"  Sentiment: {result['sentiment_score']:.3f}")
