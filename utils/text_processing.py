"""
Text Processing Utilities for Grievance Data
Handles text cleaning, tokenization, and basic NLP operations
"""

import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from textblob import TextBlob
import warnings

warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords', quiet=True)


class TextProcessor:
    """Process and clean complaint text data"""
    
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """
        Clean and normalize text
        - Remove URLs
        - Remove special characters
        - Convert to lowercase
        - Remove extra whitespace
        """
        if not isinstance(text, str):
            return ""
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize(self, text):
        """Tokenize text into words"""
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """Remove common stopwords"""
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def get_sentiment(self, text):
        """
        Analyze sentiment of text using TextBlob
        Returns polarity score (-1 to 1) and subjectivity (0 to 1)
        """
        try:
            blob = TextBlob(text)
            return {
                'polarity': blob.sentiment.polarity,
                'subjectivity': blob.sentiment.subjectivity
            }
        except:
            return {'polarity': 0.0, 'subjectivity': 0.5}
    
    def extract_keywords(self, text, top_n=5):
        """Extract top keywords from text"""
        tokens = self.tokenize(self.clean_text(text))
        tokens = self.remove_stopwords(tokens)
        
        # Count frequency
        freq_dist = nltk.FreqDist(tokens)
        
        return freq_dist.most_common(top_n)
    
    def get_text_statistics(self, text):
        """Calculate text statistics"""
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        
        return {
            'text_length': len(text),
            'cleaned_length': len(cleaned),
            'word_count': len(tokens),
            'unique_words': len(set(tokens)),
            'avg_word_length': sum(len(w) for w in tokens) / len(tokens) if tokens else 0,
            'sentence_count': len(nltk.sent_tokenize(text))
        }
    
    def process_batch(self, texts):
        """Process multiple texts and return cleaned versions"""
        return [self.clean_text(text) for text in texts]
