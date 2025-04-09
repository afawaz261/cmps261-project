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


def evaluate_model_nn(model, X_val, y_val, le, one_hot=True):
    # Predict probabilities or logits
    y_val_probs = model.predict(X_val)

    # Convert to class labels
    y_val_pred = np.argmax(y_val_probs, axis=1)
    
    # If true labels are one-hot encoded, convert them to class indices
    if one_hot:
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
    plot_confusion_matrix_nn(y_val_true, y_val_pred, le)


def plot_confusion_matrix_nn(y_true, y_pred, le):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

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
    
nlp = spacy.load('en_core_web_sm')

def preprocess(text):
    doc = nlp(str(text).lower())
    tokens = [
        token.lemma_ for token in doc
        if token.is_alpha and
           token.text not in STOP_WORDS and
           not token.is_space and
           len(token.text) > 2
    ]
    return " ".join(tokens)

def count_clean_words(status, df):
    texts = df[df['Status'].str.strip().str.lower() == status.lower()]['CleanText'].dropna()
    words = []

    for text in texts:
        for token in text.split():  # already lemmatized words are in CleanText
            words.append(token)

    return Counter(words).most_common(10)

def get_clean_word_freq(df):
    statuses = df['Status'].dropna().unique()
    for status in statuses:
        print(f"\nTop words in {status}:")
        print(count_clean_words(status, df))
        
def visualize_word_clouds(df):
    statuses = df['Status'].dropna().unique()
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for i, label in enumerate(statuses):
        # Get all text from that label
        texts = df[df['Status'].str.strip().str.lower() == label.lower()]['Statement'].dropna()
        full_text = " ".join(texts.astype(str)).lower()

        # Generate WordCloud
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            stopwords=STOP_WORDS,
            collocations=True
        ).generate(full_text)

        # Plot it
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].axis("off")
        axes[i].set_title(f"{label}", fontsize=14)

# Hide empty subplot (8th box)
    axes[-1].axis("off")

    plt.tight_layout()
    plt.show()

def extract_features(df):
    # Initialize TF-IDF Vectorizer
    tfidf = TfidfVectorizer(max_features=5000)
    
    # Fit and transform the CleanText
    X_tfidf = tfidf.fit_transform(df['CleanText'])
    
    return X_tfidf, tfidf


def encode_labels(df):
    # Encode the 'Status' labels
    df['Status'] = df['Status'].dropna()
    le = LabelEncoder()
    y = le.fit_transform(df['Status'])
    
    return y, le

def split_data(X, y):
    # Split the data into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)
    
    print(f"Train size: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def handle_imbalance(X_train, y_train):
    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    df.columns = ['Statement', 'Status']
    df = df.dropna(subset=['Statement'])
    df = df.drop_duplicates()
    return df

def check_class_distribution(df):
    status_counts = df['Status'].value_counts()
    print("Class Distribution:\n", status_counts)
    
    df['Status'].value_counts().plot(kind='bar')
    plt.title("Class Distribution")
    plt.xlabel("Labels")
    plt.ylabel("Counts")
    plt.show()
    
def add_text_length(df):
    # Add a column for text length (in words)
    df['TextLength'] = df['Statement'].apply(lambda x: len(str(x).split()))
    
    return df

def check_text_length_distribution(df):
    df['TextLength'].hist(bins=50)
    plt.title("Distribution of Text Lengths")  
    plt.show() 