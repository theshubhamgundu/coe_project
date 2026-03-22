# Copilot Instructions for Grievance Classification Project

This is a comprehensive Machine Learning + Data Science project for Public Grievance Classification and Priority Prediction.

## Project Type
Python Data Science Project with ML models, EDA, and visualization dashboard

## Key Directories
- `data/` - Raw and processed datasets
- `notebooks/` - Jupyter notebooks for EDA and exploration
- `scripts/` - Python pipeline scripts (preprocessing, training, evaluation)
- `models/` - Saved trained models
- `visualizations/` - Generated plots and dashboards
- `utils/` - Helper functions and utilities

## Technology Stack
- **Data Processing**: pandas, numpy
- **ML Frameworks**: scikit-learn, xgboost
- **NLP**: nltk, textblob
- **Visualization**: matplotlib, seaborn, plotly
- **Notebooks**: jupyter, ipython

## Core Components
1. Data Preprocessing - Text cleaning, tokenization, feature extraction
2. EDA - Statistical analysis, pattern discovery
3. Feature Engineering - TF-IDF, sentiment analysis, categorical features
4. Model Training - Logistic Regression, Naive Bayes, Random Forest, XGBoost
5. Evaluation - Accuracy, Precision, Recall, F1-Score, visualization
6. Dashboard - Interactive visualizations and insights

## Key Features Implemented
- Text-based feature engineering (TF-IDF, keyword frequency)
- Time-based trend analysis
- Location-based patterns
- Sentiment analysis for priority prediction
- Comprehensive model evaluation
- Interactive visualization dashboard

## Running the Project
1. Install dependencies: `pip install -r requirements.txt`
2. Generate sample data: `python scripts/generate_sample_data.py`
3. Preprocess data: `python scripts/data_preprocessing.py`
4. Train models: `python scripts/model_training.py`
5. Evaluate and visualize: `python scripts/evaluation.py`

## Output
Each grievance receives:
- Predicted category
- Priority level (High/Medium/Low)
- Feature-based reasoning
