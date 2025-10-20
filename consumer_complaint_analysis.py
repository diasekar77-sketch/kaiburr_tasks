import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class ConsumerComplaintAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with data path"""
        self.data_path = data_path
        self.df = None
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
    def load_data(self):
        """Load and preprocess the consumer complaint data"""
        print("Loading consumer complaint data...")
        try:
            # Load the data (assuming CSV format)
            self.df = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.df.shape}")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def exploratory_data_analysis(self):
        """Perform exploratory data analysis"""
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Columns: {list(self.df.columns)}")
        
        # Check for missing values
        print("\nMissing values:")
        print(self.df.isnull().sum())
        
        # Target variable distribution
        if 'Product' in self.df.columns:
            print("\nProduct distribution:")
            print(self.df['Product'].value_counts())
            
        # Text length analysis
        if 'Consumer complaint narrative' in self.df.columns:
            self.df['text_length'] = self.df['Consumer complaint narrative'].str.len()
            print(f"\nText length statistics:")
            print(self.df['text_length'].describe())
            
        # Visualizations
        plt.figure(figsize=(15, 10))
        
        # Product distribution
        plt.subplot(2, 3, 1)
        if 'Product' in self.df.columns:
            self.df['Product'].value_counts().plot(kind='bar')
            plt.title('Product Distribution')
            plt.xticks(rotation=45)
        
        # Text length distribution
        plt.subplot(2, 3, 2)
        plt.hist(self.df['text_length'], bins=50)
        plt.title('Text Length Distribution')
        plt.xlabel('Character Count')
        
        # Company distribution (top 10)
        plt.subplot(2, 3, 3)
        if 'Company' in self.df.columns:
            self.df['Company'].value_counts().head(10).plot(kind='bar')
            plt.title('Top 10 Companies')
            plt.xticks(rotation=45)
        
        # State distribution (top 10)
        plt.subplot(2, 3, 4)
        if 'State' in self.df.columns:
            self.df['State'].value_counts().head(10).plot(kind='bar')
            plt.title('Top 10 States')
            plt.xticks(rotation=45)
        
        # Issue distribution (top 10)
        plt.subplot(2, 3, 5)
        if 'Issue' in self.df.columns:
            self.df['Issue'].value_counts().head(10).plot(kind='bar')
            plt.title('Top 10 Issues')
            plt.xticks(rotation=45)
        
        # Submission method
        plt.subplot(2, 3, 6)
        if 'Submitted via' in self.df.columns:
            self.df['Submitted via'].value_counts().plot(kind='pie', autopct='%1.1f%%')
            plt.title('Submission Method')
        
        plt.tight_layout()
        plt.show()
        
    def preprocess_text(self, text):
        """Preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = str(text).lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        
        return ' '.join(tokens)
    
    def text_preprocessing(self):
        """Apply text preprocessing to the dataset"""
        print("\n=== TEXT PREPROCESSING ===")
        
        if 'Consumer complaint narrative' in self.df.columns:
            print("Preprocessing consumer complaint narratives...")
            self.df['processed_text'] = self.df['Consumer complaint narrative'].apply(self.preprocess_text)
            
            # Remove rows with empty processed text
            self.df = self.df[self.df['processed_text'].str.len() > 0]
            print(f"After preprocessing: {self.df.shape[0]} records")
        
        # Create target variable mapping
        product_mapping = {
            'Credit reporting, repair, or other': 0,
            'Debt collection': 1,
            'Consumer Loan': 2,
            'Mortgage': 3
        }
        
        if 'Product' in self.df.columns:
            self.df['target'] = self.df['Product'].map(product_mapping)
            # Remove rows with unmapped products
            self.df = self.df.dropna(subset=['target'])
            print(f"Target distribution:")
            print(self.df['target'].value_counts())
    
    def feature_engineering(self):
        """Create additional features"""
        print("\n=== FEATURE ENGINEERING ===")
        
        # Text-based features
        if 'processed_text' in self.df.columns:
            self.df['word_count'] = self.df['processed_text'].str.split().str.len()
            self.df['char_count'] = self.df['processed_text'].str.len()
            self.df['avg_word_length'] = self.df['char_count'] / self.df['word_count']
            
        # Sentiment-related keywords
        negative_keywords = ['problem', 'issue', 'error', 'wrong', 'bad', 'terrible', 'awful', 'horrible']
        positive_keywords = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic']
        
        if 'processed_text' in self.df.columns:
            self.df['negative_sentiment'] = self.df['processed_text'].apply(
                lambda x: sum(1 for word in negative_keywords if word in x)
            )
            self.df['positive_sentiment'] = self.df['processed_text'].apply(
                lambda x: sum(1 for word in positive_keywords if word in x)
            )
    
    def train_models(self):
        """Train multiple classification models"""
        print("\n=== MODEL TRAINING ===")
        
        # Prepare data
        X = self.df['processed_text']
        y = self.df['target']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)
        
        # Define models
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42),
            'Naive Bayes': MultinomialNB()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Train model
            model.fit(X_train_tfidf, y_train)
            
            # Predictions
            y_pred = model.predict(X_test_tfidf)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train_tfidf, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'predictions': y_pred,
                'y_test': y_test
            }
            
            print(f"Accuracy: {accuracy:.4f}")
            print(f"CV Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        self.results = results
        self.tfidf = tfidf
        self.X_test = X_test
        self.y_test = y_test
        
        return results
    
    def model_evaluation(self):
        """Evaluate and compare models"""
        print("\n=== MODEL EVALUATION ===")
        
        # Create comparison plot
        model_names = list(self.results.keys())
        accuracies = [self.results[name]['accuracy'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        x = np.arange(len(model_names))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(x - width/2, accuracies, width, label='Test Accuracy', alpha=0.8)
        bars2 = ax.bar(x + width/2, cv_means, width, label='CV Mean', alpha=0.8, yerr=cv_stds, capsize=5)
        
        ax.set_xlabel('Models')
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(model_names, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.show()
        
        # Print detailed results
        print("\nDetailed Results:")
        for name, result in self.results.items():
            print(f"\n{name}:")
            print(f"  Test Accuracy: {result['accuracy']:.4f}")
            print(f"  CV Mean: {result['cv_mean']:.4f}")
            print(f"  CV Std: {result['cv_std']:.4f}")
            
            # Classification report
            print(f"\n  Classification Report:")
            print(classification_report(result['y_test'], result['predictions']))
    
    def predict_new_complaint(self, complaint_text):
        """Predict category for a new complaint"""
        if not hasattr(self, 'results') or not hasattr(self, 'tfidf'):
            print("Please train models first!")
            return None
        
        # Preprocess the new text
        processed_text = self.preprocess_text(complaint_text)
        
        # Transform using trained TF-IDF
        text_tfidf = self.tfidf.transform([processed_text])
        
        # Get predictions from all models
        predictions = {}
        for name, result in self.results.items():
            pred = result['model'].predict(text_tfidf)[0]
            predictions[name] = pred
        
        # Return predictions
        return predictions
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting Consumer Complaint Analysis...")
        
        if not self.load_data():
            return
        
        self.exploratory_data_analysis()
        self.text_preprocessing()
        self.feature_engineering()
        self.train_models()
        self.model_evaluation()
        
        print("\nAnalysis completed successfully!")

# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = ConsumerComplaintAnalyzer('consumer_complaints.csv')
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    # Example prediction
    sample_complaint = """
    I have been trying to get my credit report fixed for months. 
    There are errors on my report that are affecting my ability to get a loan.
    The credit bureau has not responded to my disputes.
    """
    
    print("\n=== SAMPLE PREDICTION ===")
    predictions = analyzer.predict_new_complaint(sample_complaint)
    print(f"Predictions for sample complaint: {predictions}")

