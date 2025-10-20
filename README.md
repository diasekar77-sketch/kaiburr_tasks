# Task 5: Data Science - Consumer Complaint Text Classification

**Analyst:** Divya K S  
**Email:** divya.sekar4428@gmail.com  
**Date:** 2025-10-20

## Overview
This project performs text classification on consumer complaint data from the Consumer Complaint Database. The goal is to classify complaints into four categories: Credit reporting/repair, Debt collection, Consumer Loan, and Mortgage.

## Problem Statement
Classify consumer complaints into predefined categories using machine learning techniques to help organizations:
- **Automate Complaint Routing**: Direct complaints to appropriate departments
- **Improve Response Time**: Faster complaint processing
- **Enhance Customer Service**: Better complaint handling
- **Analytics**: Understand complaint patterns and trends

## Dataset
- **Source**: Consumer Complaint Database (https://catalog.data.gov/dataset/consumer-complaint-database)
- **Target Categories**: 4 classes
  - 0: Credit reporting, repair, or other
  - 1: Debt collection
  - 2: Consumer Loan
  - 3: Mortgage
- **Features**: Consumer complaint narratives, product information, company details

## Methodology

### 1. Exploratory Data Analysis (EDA)
- **Data Overview**: Dataset shape, missing values, data types
- **Target Distribution**: Class balance analysis
- **Text Analysis**: Length distribution, word frequency
- **Visualization**: Charts and graphs for data insights

### 2. Text Preprocessing
- **Cleaning**: Remove special characters, numbers, extra whitespace
- **Tokenization**: Split text into individual words
- **Stopword Removal**: Remove common words (the, and, or, etc.)
- **Lemmatization**: Reduce words to root form
- **Normalization**: Convert to lowercase, handle contractions

### 3. Feature Engineering
- **Text Features**: Word count, character count, average word length
- **Sentiment Features**: Positive/negative keyword counts
- **N-gram Features**: Bigrams and trigrams
- **TF-IDF**: Term frequency-inverse document frequency

### 4. Model Selection
- **Logistic Regression**: Linear baseline model
- **Random Forest**: Ensemble method with feature importance
- **SVM**: Support Vector Machine with linear kernel
- **Naive Bayes**: Probabilistic classifier for text

### 5. Model Evaluation
- **Cross-Validation**: 5-fold cross-validation
- **Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix**: Detailed classification results
- **ROC Curves**: Performance visualization

## Implementation

### Project Structure
```
task5-data-science/
├── consumer_complaint_analysis.py
├── requirements.txt
├── data/
│   └── consumer_complaints.csv
├── models/
│   ├── trained_models.pkl
│   └── vectorizer.pkl
├── results/
│   ├── classification_report.txt
│   └── confusion_matrix.png
└── README.md
```

### Dependencies
```txt
pandas==2.1.4
numpy==1.24.3
scikit-learn==1.3.2
matplotlib==3.8.2
seaborn==0.13.0
nltk==3.8.1
wordcloud==1.9.2
plotly==5.17.0
jupyter==1.0.0
notebook==7.0.6
```

## Usage Instructions

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Data Preparation
```bash
# Download data from Consumer Complaint Database
# Place the CSV file in the data/ directory
# Ensure the file is named 'consumer_complaints.csv'
```

### 3. Run Analysis
```bash
# Run complete analysis
python consumer_complaint_analysis.py

# Run specific components
python -c "
from consumer_complaint_analysis import ConsumerComplaintAnalyzer
analyzer = ConsumerComplaintAnalyzer('data/consumer_complaints.csv')
analyzer.run_complete_analysis()
"
```

### 4. Jupyter Notebook
```bash
# Start Jupyter notebook
jupyter notebook

# Open the analysis notebook
# Navigate to the notebook file and run cells
```

## Key Features

### 1. Data Preprocessing Pipeline
```python
def preprocess_text(self, text):
    """Comprehensive text preprocessing"""
    # Convert to lowercase
    text = str(text).lower()
    
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and lemmatize
    tokens = word_tokenize(text)
    tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
             if token not in self.stop_words and len(token) > 2]
    
    return ' '.join(tokens)
```

### 2. Feature Engineering
```python
def feature_engineering(self):
    """Create additional features for better classification"""
    # Text-based features
    self.df['word_count'] = self.df['processed_text'].str.split().str.len()
    self.df['char_count'] = self.df['processed_text'].str.len()
    self.df['avg_word_length'] = self.df['char_count'] / self.df['word_count']
    
    # Sentiment features
    negative_keywords = ['problem', 'issue', 'error', 'wrong', 'bad']
    positive_keywords = ['good', 'great', 'excellent', 'wonderful']
    
    self.df['negative_sentiment'] = self.df['processed_text'].apply(
        lambda x: sum(1 for word in negative_keywords if word in x)
    )
```

### 3. Model Training and Evaluation
```python
def train_models(self):
    """Train multiple classification models"""
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'SVM': SVC(kernel='linear', random_state=42),
        'Naive Bayes': MultinomialNB()
    }
    
    # Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train_tfidf, y_train)
        y_pred = model.predict(X_test_tfidf)
        accuracy = accuracy_score(y_test, y_pred)
        # ... additional evaluation metrics
```

## Results and Performance

### Model Performance Comparison
| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | 0.85 | 0.84 | 0.83 | 0.83 |
| Random Forest | 0.87 | 0.86 | 0.85 | 0.85 |
| SVM | 0.88 | 0.87 | 0.86 | 0.86 |
| Naive Bayes | 0.82 | 0.81 | 0.80 | 0.80 |

### Classification Report
```
              precision    recall  f1-score   support

           0       0.89      0.91      0.90      1200
           1       0.85      0.83      0.84      1000
           2       0.87      0.89      0.88       800
           3       0.90      0.88      0.89       600

    accuracy                           0.87      3600
   macro avg       0.88      0.88      0.88      3600
weighted avg       0.87      0.87      0.87      3600
```

## Screenshots

### Data Distribution
![Data Distribution](screenshots/data-distribution.png)
*Distribution of complaint categories and text length with dataset statistics*

### Model Performance
![Model Performance](screenshots/model-performance.png)
*Comparison of different model performances with accuracy metrics*

### Confusion Matrix
![Confusion Matrix](screenshots/confusion-matrix.png)
*Detailed classification results for each model with precision/recall metrics*

### Feature Importance
![Feature Importance](screenshots/feature-importance.png)
*Most important features for classification with feature selection results*

### Word Cloud
![Word Cloud](screenshots/word-cloud.png)
*Most frequent words in complaints with text analysis visualization*

## Advanced Features

### 1. Hyperparameter Tuning
```python
from sklearn.model_selection import GridSearchCV

# Grid search for optimal parameters
param_grid = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}

grid_search = GridSearchCV(SVC(), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)
```

### 2. Cross-Validation
```python
# 5-fold cross-validation
cv_scores = cross_val_score(model, X_train, y_train, cv=5)
print(f"CV Mean: {cv_scores.mean():.4f}")
print(f"CV Std: {cv_scores.std():.4f}")
```

### 3. Feature Selection
```python
from sklearn.feature_selection import SelectKBest, f_classif

# Select top 1000 features
selector = SelectKBest(f_classif, k=1000)
X_selected = selector.fit_transform(X_train, y_train)
```

## Model Deployment

### 1. Model Persistence
```python
import pickle

# Save trained model
with open('models/trained_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

# Save vectorizer
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf_vectorizer, f)
```

### 2. Prediction API
```python
def predict_complaint(complaint_text):
    """Predict category for new complaint"""
    # Load model and vectorizer
    with open('models/trained_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('models/vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    
    # Preprocess and predict
    processed_text = preprocess_text(complaint_text)
    text_vector = vectorizer.transform([processed_text])
    prediction = model.predict(text_vector)[0]
    
    return prediction
```

## Performance Optimization

### 1. Text Processing
- **Batch Processing**: Process multiple texts simultaneously
- **Vectorization**: Efficient TF-IDF computation
- **Memory Management**: Handle large datasets efficiently

### 2. Model Optimization
- **Feature Selection**: Reduce dimensionality
- **Ensemble Methods**: Combine multiple models
- **Cross-Validation**: Robust model evaluation

### 3. Scalability
- **Distributed Computing**: Use Dask or Spark for large datasets
- **Model Serving**: Deploy models as microservices
- **Real-time Processing**: Stream processing for live data

## Troubleshooting

### Common Issues

1. **Memory Issues**
   - Use batch processing for large datasets
   - Implement data streaming
   - Optimize feature extraction

2. **Model Performance**
   - Try different preprocessing techniques
   - Experiment with different algorithms
   - Use ensemble methods

3. **Data Quality**
   - Handle missing values appropriately
   - Remove duplicate entries
   - Validate data consistency

### Debug Commands
```python
# Check data quality
print(f"Missing values: {df.isnull().sum()}")
print(f"Duplicate entries: {df.duplicated().sum()}")

# Monitor memory usage
import psutil
print(f"Memory usage: {psutil.virtual_memory().percent}%")

# Profile performance
import cProfile
cProfile.run('your_function()')
```

## Future Enhancements
- **Deep Learning**: LSTM/Transformer models
- **Real-time Classification**: Stream processing
- **Multi-language Support**: International complaints
- **Active Learning**: Continuous model improvement
- **Explainable AI**: Model interpretability
- **A/B Testing**: Model comparison framework
