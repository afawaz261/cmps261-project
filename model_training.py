from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

def train_model(X_train, y_train):
    # Train a Logistic Regression model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    return lr

def evaluate_model(model, X_val, y_val):
    # Evaluate the model on validation data
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    print(f"Validation Accuracy: {accuracy}")
    print(classification_report(y_val, y_val_pred))

def save_model(model, filename):
    # Save the trained model
    joblib.dump(model, filename)