# 🏗️ Project Architecture

## System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                     INPUT LAYER                                 │
│         Grievance/Complaint Text + Metadata                     │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                 PREPROCESSING LAYER                             │
├──────────────────────────────────────────────────────────────────┤
│  • Text Cleaning (remove special chars, lowercase)              │
│  • Tokenization                                                 │
│  • Stopword Removal                                             │
│  • Handle Missing Values                                        │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│              FEATURE ENGINEERING LAYER                          │
├──────────────────────────────────────────────────────────────────┤
│  Text Features:        Time Features:      Categorical:         │
│  • Length              • Hour of day       • Department          │
│  • Word count          • Day of week       • Type (encoded)      │
│  • TF-IDF              • Month/Quarter     • Location            │
│  • Sentiment           • Is weekend       • Complexity          │
│                                                                   │
│  Location Features:    Statistical:                             │
│  • Density             • Avg word length                        │
│  • Top location        • Unique words ratio                     │
│  • Count               • Uppercase ratio                        │
│                        • Special chars count                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│           FEATURE SELECTION & NORMALIZATION                     │
├──────────────────────────────────────────────────────────────────┤
│  • Correlation Analysis                                         │
│  • Feature Scaling (0-1, Standardization)                       │
│  • Feature Selection (Top K)                                    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
          ┌────────────┴──────────────┐
          │                           │
┌─────────▼───────────┐     ┌────────▼──────────┐
│ TRAIN-TEST SPLIT    │     │   STRATIFIED      │
│   80% Train         │     │   (By Category)   │
│   20% Test          │     │   (By Priority)   │
└─────────┬───────────┘     └────────┬──────────┘
          │                           │
          └────────────┬──────────────┘
                       │
      ┌────────────────┬────────────────┐
      │                │                │
┌─────▼─────────┐  ┌──▼────────────┐  ┌──▼──────────────┐
│ CLASSIFICATION│  │ CLASSIFICATION│  │ CLASSIFICATION  │
│   MODELS      │  │   MODELS      │  │    MODELS       │
│               │  │               │  │                 │
│ 1. Logistic   │  │ 1. Logistic   │  │ 1. Random       │
│    Regression │  │    Regression │  │    Forest       │
│ 2. Naive Bayes│  │ 2. Random     │  │ 2. XGBoost      │
│ 3. Random     │  │    Forest     │  │                 │
│    Forest     │  │ 3. XGBoost    │  │ (Priority       │
│ 4. XGBoost    │  │               │  │  Prediction)    │
│               │  │ (CATEGORY     │  │                 │
│ (CATEGORY     │  │  PREDICTION)  │  │                 │
│  PREDICTION)  │  │               │  │                 │
└─────┬─────────┘  └──┬────────────┘  └──┬──────────────┘
      │               │                   │
      └───────────────┬───────────────────┘
                      │
           ┌──────────▼──────────┐
           │  MODEL EVALUATION   │
           ├─────────────────────┤
           │  Accuracy           │
           │  Precision          │
           │  Recall             │
           │  F1-Score           │
           │  Confusion Matrix   │
           │  ROC-AUC Curves     │
           └──────────┬──────────┘
                      │
           ┌──────────▼──────────┐
           │ SELECT BEST MODELS  │
           │ (By F1-Score)       │
           └──────────┬──────────┘
                      │
      ┌───────────────┴───────────────┐
      │                               │
┌─────▼────────────┐         ┌───────▼──────────┐
│  PRODUCTION MODEL│         │  VISUALIZATION   │
│  (Classification)│         │  DASHBOARD       │
│  (Priority Pred.)│         │                  │
└─────┬────────────┘         │  • Category      │
      │                      │    Distribution  │
      │                      │  • Priority      │
      │                      │    Distribution  │
      │                      │  • Heatmaps      │
      │                      │  • Confusion     │
      │                      │    Matrices      │
      │                      │  • Feature       │
      │                      │    Importance    │
      │                      └───────┬──────────┘
      │                              │
┌─────▼──────────────────────────────▼──────────┐
│         PREDICTION OUTPUT LAYER                │
├───────────────────────────────────────────────┤
│  ✓ Predicted Category                         │
│  ✓ Priority Level (High/Medium/Low)           │
│  ✓ Confidence Score                           │
│  ✓ Sentiment Analysis                         │
│  ✓ Feature Contribution                       │
└───────────────────────────────────────────────┘
```

## Data Flow

### 1. Data Collection
- Public grievance data from various sources
- Includes: text, category, location, timestamp, department

### 2. Preprocessing
- Text cleaning (lowercase, remove special chars)
- Handle missing values
- Format standardization

### 3. Feature Engineering
- Extract 40+ features from complaint data
- Combine text, time, location, categorical features

### 4. Model Training
- Train 7 different ML models
- Both for category classification and priority prediction
- Use stratified train-test split (80-20)

### 5. Model Evaluation
- Evaluate using multiple metrics
- Select best model by F1-Score
- Generate confusion matrices and reports

### 6. Prediction
- New grievances fed through pipeline
- Category and priority predicted
- Confidence scores provided

### 7. Visualization & Insights
- Distribution analysis
- Trend identification
- Pattern discovery

---

## Technology Stack

### Core Libraries
```
pandas              - Data manipulation
numpy               - Numerical computing
scikit-learn        - Machine learning (models, metrics)
xgboost             - Gradient boosting
```

### NLP & Text Processing
```
nltk                - Natural language processing
textblob            - Sentiment analysis
```

### Visualization
```
matplotlib          - Static plots
seaborn             - Statistical visualizations
plotly              - Interactive dashboards
```

### Development
```
jupyter             - Interactive notebooks
ipython             - Enhanced Python shell
```

---

## Model Comparison Matrix

| Model | Speed | Accuracy | Interpretability | Best For |
|-------|-------|----------|------------------|----------|
| Logistic Regression | ⚡⚡⚡ | ⭐⭐ | ⭐⭐⭐ | Baseline, Linear relationships |
| Naive Bayes | ⚡⚡⚡ | ⭐⭐ | ⭐⭐ | Text classification |
| Random Forest | ⚡⚡ | ⭐⭐⭐ | ⭐⭐ | Balanced performance |
| XGBoost | ⚡ | ⭐⭐⭐⭐ | ⭐ | Best accuracy, slow |

---

## Feature Importance Hierarchy

### Tier 1 (Most Important)
- Sentiment polarity
- Complaint text length
- Word count
- Time of day

### Tier 2 (Important)
- Location density
- Day of week
- TF-IDF features
- Sentiment subjectivity

### Tier 3 (Supplementary)
- Department category
- Special character count
- Uppercase ratio
- Complaint status

---

## Performance Metrics

### Target Metric: F1-Score
- Balances precision and recall
- Suitable for classification
- Weights: precision=0.5, recall=0.5

### Why F1-Score?
- ✓ Handles class imbalance
- ✓ Considers false positives and negatives
- ✓ Single metric for multiple classes

---

## Scalability Considerations

### Current Capacity
- **Data:** Up to 100K records
- **Features:** 40+ numeric features
- **Models:** 4+ ensemble models
- **Speed:** ~100 predictions/second

### Scaling Tips
1. Use distributed computing (Spark) for 1M+ records
2. Implement model serving (Flask/FastAPI)
3. Use cloud platforms (AWS, GCP, Azure)
4. Implement caching for repeated predictions
5. Use model compression (quantization)

---

## Future Enhancements

### Short-term
- [ ] Add deep learning models (LSTM, BERT)
- [ ] Implement cross-validation
- [ ] Add hyperparameter optimization

### Medium-term
- [ ] Deploy as REST API
- [ ] Add real-time prediction dashboard
- [ ] Implement model monitoring

### Long-term
- [ ] Federated learning
- [ ] AutoML pipeline
- [ ] Multi-modal analysis (text + images)

