# 🎯 QUICK START GUIDE
## Public Grievance Classification & Priority Prediction System

### 📦 Installation

**1. Create virtual environment (recommended):**
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```

---

### 🚀 Running the Project

#### **Option 1: Quick Pipeline (Recommended)**
Run all steps in sequence:
```bash
python scripts/run_pipeline.py
```

This will:
1. ✓ Generate sample data (1000 grievance records)
2. ✓ Preprocess and engineer features
3. ✓ Train 4 classification + 3 priority models
4. ✓ Evaluate models and create visualizations

**Time:** ~2-5 minutes

---

#### **Option 2: Step-by-Step

**Step 1: Generate Sample Data**
```bash
python scripts/generate_sample_data.py
```
Output: `data/grievances_dataset.csv` (1000 sample records)

**Step 2: Preprocess Data & Engineer Features**
```bash
python scripts/data_preprocessing.py
```
Output: `data/grievances_processed.csv` (processed features)

**Step 3: Train Models**
```bash
python scripts/model_training.py
```
Output: Models in `models/` directory

**Step 4: Evaluate & Visualize**
```bash
python scripts/evaluation.py
```
Output: Charts in `visualizations/` directory

---

#### **Option 3: Interactive Exploration**
**Launch Jupyter notebook:**
```bash
jupyter notebook notebooks/EDA.ipynb
```

This notebook includes:
- Data loading and exploration
- Text cleaning and preprocessing
- EDA visualizations
- Model training and comparison
- Prediction examples

---

### 📊 Project Structure

```
coe_project/
├── data/                    # Datasets (raw & processed)
├── notebooks/               # Jupyter notebooks
│   └── EDA.ipynb           # Interactive exploration
├── scripts/                 # Pipeline scripts
│   ├── generate_sample_data.py      # Create sample data
│   ├── data_preprocessing.py        # Feature engineering
│   ├── model_training.py            # Train models
│   ├── evaluation.py                # Evaluate & visualize
│   ├── prediction_engine.py         # Make predictions
│   └── run_pipeline.py              # Run all steps
├── models/                  # Trained models (.pkl files)
├── visualizations/          # Generated plots & dashboards
├── utils/                   # Helper functions
│   ├── text_processing.py   # NLP utilities
│   └── feature_engineering.py # Feature extraction
├── requirements.txt         # Python dependencies
└── README.md               # Project documentation
```

---

### 📈 Expected Outputs

After running the pipeline, you'll have:

**1. Data Files:**
- `data/grievances_dataset.csv` - Raw data (1000 records)
- `data/grievances_processed.csv` - Engineered features

**2. Models:**
- `models/classification_*.pkl` - Category classifiers
- `models/priority_*.pkl` - Priority predictors
- `models/label_mappings.pkl` - Encoding mappings

**3. Visualizations:**
- `visualizations/category_distribution.html` - Interactive charts
- `visualizations/priority_distribution.html` - Priority breakdown
- `visualizations/location_distribution.html` - Location analysis
- `visualizations/category_priority_heatmap.html` - Correlation heatmap
- `visualizations/confusion_matrix_*.png` - Model performance
- `visualizations/classification_report_*.png` - Metrics
- `visualizations/feature_importance_*.png` - Top features

---

### 🔮 Making Predictions

**Use the prediction engine:**

```python
from scripts.prediction_engine import GrievancePredictionEngine

# Load engine
engine = GrievancePredictionEngine()

# Make prediction
result = engine.predict("Deep pothole on Main Road causing accidents")

print(f"Category: {result['predicted_category']}")
print(f"Priority: {result['priority_level']}")
print(f"Confidence: {result['category_confidence']:.2%}")
```

Or run the built-in test:
```bash
python scripts/prediction_engine.py
```

---

### 📊 Understanding the Models

**Classification Models:**
- **Logistic Regression** - Baseline, fast
- **Naive Bayes** - Good for text data
- **Random Forest** - Best balance of accuracy & speed
- **XGBoost** - Highest accuracy, slower

**Metrics Used:**
- **Accuracy** - Overall correctness
- **Precision** - Correctness of predictions
- **Recall** - Coverage of actual cases
- **F1-Score** - Harmonic mean (primary metric)

---

### 🧠 Key Features Engineered

**Text Features:**
- Complaint length, word count
- TF-IDF vectorization
- Sentiment polarity & subjectivity

**Time Features:**
- Hour of day, day of week
- Month, quarter
- Weekend flag

**Location Features:**
- Complaint count per location
- Location density
- Top location indicator

**Categorical Features:**
- Department (one-hot encoded)
- Complaint type categories

---

### 🐛 Troubleshooting

**Issue:** NLTK data missing
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

**Issue:** Out of memory
- Reduce dataset size in `generate_sample_data.py`
- Use fewer features or smaller models

**Issue:** Models not found
- Run preprocessing and training first
- Ensure `data/grievances_processed.csv` exists

---

### 📚 Learning Resources

1. **Jupyter Notebook** - Best for learning the pipeline
2. **README.md** - Full project documentation
3. **Code Comments** - Well-documented source code
4. **Visualization Outputs** - See results in `visualizations/`

---

### ⚙️ Configuration

Edit these files to customize:

**Adjust sample data size:**
- Edit `generate_sample_data.py` - Change `n_samples` parameter

**Adjust model training:**
- Edit `model_training.py` - Modify hyperparameters

**Change features:**
- Edit `utils/feature_engineering.py` - Add/remove features

---

### 🎯 Next Steps

1. **Try the quick start:**
   ```bash
   python scripts/run_pipeline.py
   ```

2. **Explore the notebook:**
   ```bash
   jupyter notebook notebooks/EDA.ipynb
   ```

3. **Review visualizations:**
   Open files in `visualizations/` folder

4. **Make predictions:**
   ```bash
   python scripts/prediction_engine.py
   ```

5. **Use with real data:**
   Replace `data/grievances_dataset.csv` with your data

---

### 📞 Questions?

Refer to the comments in individual scripts for implementation details.

**Happy analyzing! 🚀**

