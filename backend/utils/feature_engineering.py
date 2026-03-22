"""
Feature Engineering Utilities for Grievance Data
Extracts and transforms features for machine learning models
"""

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """Extract and engineer features from grievance data"""
    
    def __init__(self):
        self.tfidf_vectorizer = None
        self.count_vectorizer = None
    
    # Text-based features
    def fit_tfidf(self, texts, max_features=100):
        """Fit TF-IDF vectorizer on texts"""
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )
        self.tfidf_vectorizer.fit(texts)
        return self
    
    def transform_tfidf(self, texts):
        """Transform texts to TF-IDF matrix"""
        if self.tfidf_vectorizer is None:
            raise ValueError("TF-IDF vectorizer not fitted. Call fit_tfidf first.")
        return self.tfidf_vectorizer.transform(texts)
    
    def fit_count_vectors(self, texts, max_features=100):
        """Fit Count vectorizer on texts"""
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            stop_words='english',
            lowercase=True,
            min_df=2,
            max_df=0.8
        )
        self.count_vectorizer.fit(texts)
        return self
    
    def transform_count_vectors(self, texts):
        """Transform texts to count matrix"""
        if self.count_vectorizer is None:
            raise ValueError("Count vectorizer not fitted. Call fit_count_vectors first.")
        return self.count_vectorizer.transform(texts)
    
    # Time-based features
    @staticmethod
    def extract_time_features(timestamp_series):
        """Extract time-based features from timestamps"""
        timestamp_series = pd.to_datetime(timestamp_series)
        
        features = pd.DataFrame()
        features['hour'] = timestamp_series.dt.hour
        features['day_of_week'] = timestamp_series.dt.dayofweek
        features['day_of_month'] = timestamp_series.dt.day
        features['month'] = timestamp_series.dt.month
        features['quarter'] = timestamp_series.dt.quarter
        features['is_weekend'] = (timestamp_series.dt.dayofweek >= 5).astype(int)
        
        return features
    
    # Text statistics features
    @staticmethod
    def extract_text_stats(texts):
        """Extract statistical features from text"""
        features = pd.DataFrame()
        
        features['text_length'] = texts.str.len()
        features['word_count'] = texts.str.split().str.len()
        features['avg_word_length'] = features['text_length'] / features['word_count']
        features['unique_words_ratio'] = texts.apply(
            lambda x: len(set(x.split())) / len(x.split()) if len(x.split()) > 0 else 0
        )
        features['uppercase_ratio'] = texts.apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        features['digit_count'] = texts.str.count(r'\d')
        features['special_char_count'] = texts.str.count(r'[!?.,;:-]')
        
        return features
    
    # Categorical features encoder
    @staticmethod
    def encode_categorical(df, categorical_cols):
        """
        Encode categorical features using one-hot encoding
        """
        df_encoded = df.copy()
        
        for col in categorical_cols:
            if col in df_encoded.columns:
                dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded, dummies], axis=1)
                df_encoded = df_encoded.drop(col, axis=1)
        
        return df_encoded
    
    # Location-based features
    @staticmethod
    def extract_location_features(locations_series):
        """Extract location-based features"""
        features = pd.DataFrame()
        
        # Count complaints per location (location popularity)
        location_counts = locations_series.value_counts()
        features['location_complaint_count'] = locations_series.map(location_counts)
        
        # Location is top location (top 10)
        top_locations = location_counts.head(10).index
        features['is_top_location'] = locations_series.isin(top_locations).astype(int)
        
        # Location density
        total_complaints = len(locations_series)
        features['location_density'] = features['location_complaint_count'] / total_complaints
        
        return features
    
    # Sentiment features
    @staticmethod
    def extract_sentiment_features(sentiments):
        """Extract sentiment-based features"""
        sentiment_df = pd.DataFrame(list(sentiments))
        
        # Ensure we have the columns we need
        if 'polarity' not in sentiment_df.columns:
            # Handle case where columns might be indexed
            if len(sentiment_df.columns) >= 1:
                sentiment_df.columns = ['polarity', 'subjectivity'] if len(sentiment_df.columns) >= 2 else ['polarity']
        
        # Create sentiment categories
        sentiment_df['sentiment_category'] = pd.cut(
            sentiment_df['polarity'],
            bins=[-1, -0.1, 0.1, 1],
            labels=['negative', 'neutral', 'positive']
        )
        
        # Sentiment intensity
        sentiment_df['sentiment_intensity'] = sentiment_df['polarity'].abs()
        
        return sentiment_df


class FeatureScaler:
    """Scale and normalize features"""
    
    @staticmethod
    def normalize_features(df, exclude_cols=None):
        """
        Normalize numerical features to [0, 1] range
        """
        df_normalized = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_cols:
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if max_val > min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
        
        return df_normalized
    
    @staticmethod
    def standardize_features(df, exclude_cols=None):
        """
        Standardize features to mean=0, std=1
        """
        df_standardized = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if exclude_cols:
            numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        for col in numeric_cols:
            mean = df_standardized[col].mean()
            std = df_standardized[col].std()
            if std > 0:
                df_standardized[col] = (df_standardized[col] - mean) / std
        
        return df_standardized
