import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import numpy as np
from xgboost import XGBClassifier
import glob

def train_lr_model(X_train, y_train):
    # Train a Logistic Regression model
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    return lr

def train_rf_model(X_train, y_train):
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs= -1)
    rf.fit(X_train, y_train)
    
    return rf

def train_nb_model(X_train, y_train):
    nb = MultinomialNB()
    nb.fit(X_train, y_train)
    
    return nb

def train_cnn_model(X_train, y_train, X_val, y_val, y_test):
    y_train_cat = to_categorical(y_train)
    y_val_cat = to_categorical(y_val)
    y_test_cat = to_categorical(y_test)
    
    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(X_train.shape[1],)))
    model.add(Dropout(0.3))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(y_train_cat.shape[1], activation='softmax'))  # Output layer

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    
    history = model.fit(
        X_train.toarray(), y_train_cat,
        epochs=10,
        batch_size=64,
        validation_data=(X_val.toarray(), y_val_cat),
        verbose=2
    )
    
    return model

def train_xgb_model(X_train, y_train):
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
    xgb_model.fit(X_train, y_train)
    
    return xgb_model

def evaluate_model(model, X_val, y_val, le):
    # Evaluate the model on validation data
    y_val_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    
    print(f"Validation Accuracy: {accuracy}")
    
    print(classification_report(y_val, y_val_pred, target_names=le.classes_))
    
    report = classification_report(y_val, y_val_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['val_accuracy'] = accuracy  # Add accuracy to the report

    report_filename = f"validation_evaluation_reports/{model.__class__.__name__}_validation_evaluation_report.csv"
    
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)

    report_df.to_csv(report_filename, index=True)
    print(f"Evaluation report saved as {report_filename}")
    
    plot_confusion_matrix(model, y_val, y_val_pred, le)


def evaluate_model_nn(model, X_val, y_val, le):
    # Predict probabilities or logits
    y_val_probs = model.predict(X_val)

    # Convert to class labels
    y_val_pred = np.argmax(y_val_probs, axis=1)
        
    if y_val.ndim == 2:  
        y_val_true = np.argmax(y_val, axis=1)
    else:
        y_val_true = y_val

    # Evaluate
    accuracy = accuracy_score(y_val_true, y_val_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")
    
    print(classification_report(y_val_true, y_val_pred, target_names=le.classes_))

    # Save report
    report = classification_report(y_val_true, y_val_pred, target_names=le.classes_, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    report_df['val_accuracy'] = accuracy

    report_filename = f"validation_evaluation_reports/NeuralNetwork_validation_evaluation_report.csv"
    os.makedirs(os.path.dirname(report_filename), exist_ok=True)
    report_df.to_csv(report_filename, index=True)
    print(f"Evaluation report saved as {report_filename}")

    # Plot confusion matrix
    plot_confusion_matrix(model, y_val_true, y_val_pred, le)

def save_model(model):
    model_name = model.__class__.__name__
    
    models_path = "models"
    
    if not os.path.exists(models_path):
        os.makedirs(models_path)
    
    file_path = os.path.join(models_path, f"{model_name}.pkl")
    
    joblib.dump(model, file_path)
    print(f"Model saved as {file_path}")
    
def plot_confusion_matrix(model, y_val, y_val_pred, le):
    cm = confusion_matrix(y_val, y_val_pred)
    
    results_folder = "confusion_matrices"
    if not os.path.exists(results_folder):
        os.makedirs(results_folder)
        
    plt.figure(figsize=(10, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_,
                yticklabels=le.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"{model.__class__.__name__} - Confusion Matrix (Validation Set)")
    
    file_name = f"{model.__class__.__name__}_confusion_matrix.png"
    file_path = os.path.join(results_folder, file_name)
    plt.savefig(file_path)
    print(f"Confusion matrix saved as {file_path}")
    
    plt.show()
    
def summary_of_models():
    csv_files = glob.glob("validation_evaluation_reports/*_validation_evaluation_report.csv")
    model_summaries = []
    
    for file in csv_files:
        df = pd.read_csv(file, index_col=0)

        model_name = os.path.basename(file).replace("_validation_evaluation_report.csv", "")
        
        macro_avg = df.loc['macro avg']
        weighted_avg = df.loc['weighted avg']
        accuracy = df.loc['accuracy']['val_accuracy']  

        summary = {
            'Model': model_name,
            'Accuracy': accuracy,
            'Macro Precision': macro_avg['precision'],
            'Macro Recall': macro_avg['recall'],
            'Macro F1': macro_avg['f1-score'],
            'Weighted Precision': weighted_avg['precision'],
            'Weighted Recall': weighted_avg['recall'],
            'Weighted F1': weighted_avg['f1-score'],
        }
        model_summaries.append(summary)
    
    summary_df = pd.DataFrame(model_summaries)
    
    summary_df = summary_df.sort_values(by='Macro F1', ascending=False)
    
    summary_df.to_csv("validation_evaluation_reports/model_summary.csv", index=True)

    print(summary_df)

def load_model_if_exists(model_name):
    file_path = os.path.join("models", f"{model_name}.pkl")
    
    if os.path.exists(file_path):
        print(f"Model '{model_name}' already exists. Loading from file.")
        return joblib.load(file_path)
    else:
        return None