# Public Grievance Classification and Priority Prediction System

A comprehensive Machine Learning-based Data Science system for analyzing, classifying, and predicting priority levels of public grievances.

## 🎯 Project Objectives

- Analyze historical grievance data
- Classify complaints into categories using ML models
- Predict priority levels based on complaint features
- Provide visual insights for data-driven decision making

## 📊 Core Features

### 1. Text-Based Feature Engineering
- TF-IDF vectorization
- Bag of Words representation
- Complaint length analysis
- Keyword frequency extraction

### 2. Categorical Features
- Department classification
- Complaint type categorization

### 3. Time-Based Features
- Day/Month trend analysis
- Peak complaint hours identification

### 4. Location-Based Features
- City/District analysis
- Complaint density mapping

### 5. Sentiment Analysis
- Positive/Negative scoring
- Sentiment-based priority weighting

## 🧱 Models Implemented

1. **Logistic Regression** - Baseline classifier
2. **Naive Bayes** - Text classification specialist
3. **Random Forest** - Ensemble method
4. **XGBoost** - Advanced gradient boosting

## 📈 Visualization Dashboard

- Complaints per category
- Location heatmaps
- Time trend graphs
- Priority distribution
- Model performance charts
- Feature importance visualizations

## 📁 Project Structure

```
coe_project/
├── data/                    # Raw and processed datasets
├── notebooks/               # Jupyter notebooks for exploration
├── scripts/                 # Python scripts for pipeline
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model_training.py
│   └── evaluation.py
├── models/                  # Saved trained models
├── visualizations/          # Generated plots and dashboards
├── utils/                   # Helper functions and utilities
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## 🚀 Getting Started

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Data
Place your grievance dataset in the `data/` folder with columns:
- `complaint_text` - The grievance description
- `category` - Complaint category
- `location` - Geographic location
- `timestamp` - Date/time of complaint
- `priority` - Priority level (Target variable)

### 3. Run the Pipeline
```bash
# Generate sample data
python scripts/generate_sample_data.py

# Preprocess and feature engineering
python scripts/data_preprocessing.py

# Train models
python scripts/model_training.py

# Evaluate and generate visualizations
python scripts/evaluation.py
```

### 4. Explore Results
- Check `notebooks/EDA.ipynb` for exploratory analysis
- Review `visualizations/` folder for generated plots
- Find trained models in `models/` folder

## 📊 Output for Each Complaint

For each grievance, the system provides:
1. **Classification** - Predicted complaint category
2. **Priority Level** - High/Medium/Low
3. **Reasoning** - Feature-based explanation

## 🔄 Workflow

```
Data Collection
    ↓
Data Preprocessing
    ↓
Exploratory Data Analysis (EDA)
    ↓
Feature Engineering
    ↓
Model Training & Selection
    ↓
Model Evaluation
    ↓
Visualization & Insights
    ↓
Deployment Ready
```

## 📝 Key Dependencies

- **pandas** - Data manipulation and analysis
- **scikit-learn** - Machine learning models
- **nltk** - Natural language processing
- **matplotlib/seaborn** - Static visualizations
- **plotly** - Interactive dashboards
- **xgboost** - Gradient boosting models

## 📞 Results Metrics

The system evaluates models using:
- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC curves

## 🎓 Next Steps

1. Collect real grievance data
2. Refine feature engineering
3. Hyperparameter tuning
4. Cross-validation
5. Model deployment
6. Performance monitoring

---

**Version**: 1.0  
**Last Updated**: March 2026
