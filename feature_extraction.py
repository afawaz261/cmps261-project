
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

def extract_features(df):
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000)
    
    # Fit and transform the CleanText
    X_tfidf = tfidf.fit_transform(df['CleanText'])
    
    return X_tfidf, tfidf


def encode_labels(df):
    # Encode the 'Status' labels
    le = LabelEncoder()
    y = le.fit_transform(df['Status'])
    return y, le

def split_data(X, y):
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def handle_imbalance(X_train, y_train):
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled