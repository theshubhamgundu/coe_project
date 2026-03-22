"""
Data Preprocessing and Feature Engineering Pipeline
Processes raw grievance data for machine learning
"""

import pandas as pd
import numpy as np
import os
import sys
import warnings

warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils import TextProcessor, FeatureEngineer, FeatureScaler


def load_data(input_path='data/grievances_dataset.csv'):
    """Load raw grievance data"""
    print(f"Loading data from {input_path}...")
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data file not found: {input_path}")
    
    df = pd.read_csv(input_path)
    print(f"✓ Loaded {len(df)} records")
    
    return df


def preprocess_text(df):
    """Clean and preprocess complaint text"""
    print("\n📝 Preprocessing text data...")
    
    text_processor = TextProcessor()
    
    # Clean text
    df['complaint_text_cleaned'] = df['complaint_text'].apply(text_processor.clean_text)
    
    # Extract text statistics
    text_stats = text_processor.get_text_statistics(df['complaint_text'].iloc[0])
    print(f"✓ Text statistics calculated for each complaint")
    
    # Extract sentiment
    print("   Analyzing sentiment...")
    df['sentiment_data'] = df['complaint_text'].apply(text_processor.get_sentiment)
    df['sentiment_polarity'] = df['sentiment_data'].apply(lambda x: x['polarity'])
    df['sentiment_subjectivity'] = df['sentiment_data'].apply(lambda x: x['subjectivity'])
    
    print(f"✓ Text preprocessing complete")
    
    return df, text_processor


def extract_features(df, text_processor):
    """Extract engineered features"""
    print("\n🔧 Engineering features...")
    
    feature_engineer = FeatureEngineer()
    
    # Extract text statistics
    text_stats = FeatureEngineer.extract_text_stats(df['complaint_text_cleaned'])
    print(f"   ✓ Text statistics: {len(text_stats.columns)} features")
    
    # Extract time features
    time_features = FeatureEngineer.extract_time_features(df['timestamp'])
    print(f"   ✓ Time features: {len(time_features.columns)} features")
    
    # Extract location features
    location_features = FeatureEngineer.extract_location_features(df['location'])
    print(f"   ✓ Location features: {len(location_features.columns)} features")
    
    # Extract sentiment features
    sentiment_data = df[['sentiment_polarity', 'sentiment_subjectivity']].copy()
    sentiment_data.columns = ['polarity', 'subjectivity']
    sentiment_df = FeatureEngineer.extract_sentiment_features(
        sentiment_data.apply(lambda x: dict(x), axis=1)
    )
    print(f"   ✓ Sentiment features: {len(sentiment_df.columns)} features")
    
    # 5. Extract TF-IDF features
    # This is CRITICAL for the model to actually read the words
    print("   ✓ Fitting TF-IDF...")
    tfidf_engineer = FeatureEngineer()
    tfidf_engineer.fit_tfidf(df['complaint_text_cleaned'], max_features=100)
    tfidf_matrix = tfidf_engineer.transform_tfidf(df['complaint_text_cleaned'])
    tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), 
                           columns=[f'tfidf_{i}' for i in range(tfidf_matrix.shape[1])])
    
    # Save the vectorizer
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(tfidf_engineer.tfidf_vectorizer, 'models/tfidf_vectorizer.pkl')
    
    # Combine all features
    df_features = pd.concat([
        df[['complaint_id', 'category', 'priority', 'location', 'department']],
        text_stats,
        time_features,
        location_features,
        sentiment_df,
        tfidf_df
    ], axis=1)
    
    print(f"✓ Feature engineering complete ({df_features.shape[1]} features total)")
    
    return df_features


def encode_categorical_features(df):
    """Encode categorical variables"""
    print("\n🔠 Encoding categorical features...")
    
    df_encoded = df.copy()
    
    # Store original category and priority for use as targets
    category_orig = df_encoded['category'].copy()
    priority_orig = df_encoded['priority'].copy()
    
    # Encode categorical columns for features
    categorical_cols = ['category', 'department']
    
    # Encode department for features
    if 'department' in df_encoded.columns:
        dept_dummies = pd.get_dummies(df_encoded['department'], prefix='department', drop_first=True)
        df_encoded = pd.concat([df_encoded, dept_dummies], axis=1)
        df_encoded = df_encoded.drop('department', axis=1)
    
    # Encode sentiment category
    if 'sentiment_category' in df_encoded.columns:
        sentiment_dummies = pd.get_dummies(df_encoded['sentiment_category'], 
                                          prefix='sentiment', drop_first=True)
        df_encoded = pd.concat([df_encoded, sentiment_dummies], axis=1)
        df_encoded = df_encoded.drop('sentiment_category', axis=1) # This line was missing in the original snippet
    
    print(f"✓ Categorical encoding complete")
    
    return df_encoded


def scale_features(df):
    """Normalize numerical features"""
    print("\n📏 Scaling features...")
    
    # Identify numerical columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Don't scale the target variable
    exclude_cols = ['priority']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Normalize features
    scaler = FeatureScaler()
    df_scaled = scaler.normalize_features(df, exclude_cols=exclude_cols)
    
    print(f"✓ Feature scaling complete ({len(numeric_cols)} features scaled)")
    
    return df_scaled


def save_processed_data(df, output_path='data/grievances_processed.csv'):
    """Save processed dataset"""
    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\n✓ Processed data saved to {output_path}")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {list(df.columns)[:5]}...")


def main():
    """Run complete preprocessing pipeline"""
    print("="*60)
    print("PUBLIC GRIEVANCE DATA PREPROCESSING PIPELINE")
    print("="*60)
    
    try:
        # Load data
        df = load_data('data/grievances_dataset.csv')
        print(f"Dataset shape: {df.shape}")
        
        # Preprocess text
        df, text_processor = preprocess_text(df)
        
        # Extract features
        df_features = extract_features(df, text_processor)
        
        # Encode categorical features
        df_encoded = encode_categorical_features(df_features)
        
        # Scale features
        df_scaled = scale_features(df_encoded)
        
        # Save processed data
        save_processed_data(df_scaled)
        
        print("\n" + "="*60)
        print("✅ PREPROCESSING COMPLETE")
        print("="*60)
        print(f"Input records: {len(df)}")
        print(f"Output records: {len(df_scaled)}")
        print(f"Features engineered: {df_scaled.shape[1]}")
        
    except Exception as e:
        print(f"\n❌ Error during preprocessing: {str(e)}")
        raise


if __name__ == "__main__":
    main()
