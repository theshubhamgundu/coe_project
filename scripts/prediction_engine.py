"""
Model Prediction Engine
Provides easy interface for making predictions on new grievances
"""

import pandas as pd
import numpy as np
import joblib
import sys
import os

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
    
    def predict(self, complaint_text, location='Unknown', department='General'):
        """
        Predict category and priority for a grievance
        
        Returns:
            dict with predicted_category, priority_level, and confidence scores
        """
        
        # Process text
        cleaned_text = self.text_processor.clean_text(complaint_text)
        sentiment = self.text_processor.get_sentiment(complaint_text)
        stats = self.text_processor.get_text_statistics(complaint_text)
        
        # Create feature vector
        features = {
            'text_length': stats['text_length'],
            'word_count': stats['word_count'],
            'unique_words_ratio': stats['unique_words'],
            'avg_word_length': stats['avg_word_length'],
            'sentence_count': stats['sentence_count'],
            'sentiment_polarity': sentiment['polarity'],
            'sentiment_subjectivity': sentiment['subjectivity'],
        }
        
        # Make predictions
        category_pred = self.clf_model.predict([list(features.values())])[0]
        priority_pred = self.priority_model.predict([list(features.values())])[0]
        
        # Get confidence scores
        category_proba = self.clf_model.predict_proba([list(features.values())])[0]
        priority_proba = self.priority_model.predict_proba([list(features.values())])[0]
        
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
