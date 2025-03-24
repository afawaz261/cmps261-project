#!/usr/bin/env python
# coding: utf-8

# **CMPS 261 Project**

# In[3]:


# Importing all necessary libraries

# Data handling
import pandas as pd
import numpy as np

# Text preprocessing
import re
import string
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from collections import Counter

# Feature extraction
from sklearn.feature_extraction.text import TfidfVectorizer

# Handling imbalance
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# Evaluation
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

# Neural Network
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud


# **EXPLORATORY STAGE**

# In[4]:


# Loading the dataset

df = pd.read_csv("Sentiment_analysis_dataset.csv") 

# Quick look
df.head()
df.info()
df.describe()


# In[5]:


# Rename columns for consistency (optional)
df.columns = ['Statement', 'Status']


# In[6]:


# Drop null values in Statement
df = df.dropna(subset=['Statement'])

# Drop exact duplicates
df = df.drop_duplicates()

print(f"Dataset shape after cleaning: {df.shape}")


# In[7]:


# Check class distribution
status_counts = df['Status'].value_counts()
print("Class Distribution:\n", status_counts)


# In[8]:


# Plot class distribution
df['Status'].value_counts().plot(kind='bar')
plt.title("Class Distribution")
plt.xlabel("Labels")
plt.ylabel("Counts")


# In[9]:


# Add a column for text length (in words)
df['TextLength'] = df['Statement'].apply(lambda x: len(str(x).split()))


# In[10]:


# Text length column (optional)
df['text_length'] = df['Statement'].apply(lambda x: len(str(x).split()))  # replace 'text_column'

# Plot histogram of text lengths
df['text_length'].hist(bins=50)
plt.title("Distribution of Text Lengths")


# In[11]:


# Percentage breakdown of each class (to detect imbalance)
print("\nClass Percentages:")
print(round(df['Status'].value_counts(normalize=True) * 100, 2))


# In[12]:


# Total missing values per column
df.isnull().sum()


# In[ ]:


import spacy 
# Load SpaCy English model
nlp = spacy.load('en_core_web_sm')
nlp.max_length = 2_000_000

from spacy.lang.en.stop_words import STOP_WORDS
import string

def get_clean_word_freq(label):
    texts = df[df['Status'].str.strip().str.lower() == label.lower()]['Statement'].dropna()
    words = []

    for text in texts:
        doc = nlp(str(text).lower())
        for token in doc:
            if (token.text not in STOP_WORDS and 
                token.is_alpha and 
                not token.is_space and 
                len(token.text) > 2):
                words.append(token.lemma_)  # use lemma instead of raw word

    return Counter(words).most_common(10)

statuses = df['Status'].dropna().unique()

for status in statuses:
    print(f"\nTop words in {status}:")
    print(get_clean_word_freq(status))


# In[62]:


# List of all your sentiment labels
statuses = [
    "Normal",
    "Suicidal",
    "Depression",
    "Bipolar",
    "Anxiety",
    "Stress",
    "Personality disorder"
]

# Create subplots: 2 rows, 4 columns
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


# **PROCESSSING STAGE**

# In[15]:


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

df['CleanText'] = df['Statement'].apply(preprocess)


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize vectorizer
tfidf = TfidfVectorizer(max_features=5000)  # You can change this number

# Fit and transform
X_tfidf = tfidf.fit_transform(df['CleanText'])

# Convert to DataFrame
X_tfidf_df = pd.DataFrame(X_tfidf.toarray(), columns=tfidf.get_feature_names_out())


# In[68]:


from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
y = le.fit_transform(df['Status'])


# In[69]:


from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X_tfidf, y, test_size=0.2, stratify=y, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42)

print(f"Train size: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")


# In[70]:


from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)


# **LEARNING STAGE**

# *Logistic Regression*

# In[71]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Train
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_resampled, y_resampled)


# In[73]:


y_val_pred = lr.predict(X_val)

print("Validation Accuracy:", accuracy_score(y_val, y_val_pred))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_))


# In[74]:


import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

cm = confusion_matrix(y_val, y_val_pred)
plt.figure(figsize=(10, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Logistic Regression - Confusion Matrix (Validation Set)")
plt.show()


# *Random Forest*

# In[75]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_resampled, y_resampled)


# In[76]:


y_val_pred_rf = rf.predict(X_val)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Validation Accuracy (Random Forest):", accuracy_score(y_val, y_val_pred_rf))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_rf, target_names=le.classes_))


# In[77]:


import seaborn as sns
import matplotlib.pyplot as plt

cm_rf = confusion_matrix(y_val, y_val_pred_rf)

plt.figure(figsize=(10, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Random Forest - Confusion Matrix (Validation Set)")
plt.show()


# *Multinomial Naive Bayes*

# In[78]:


from sklearn.naive_bayes import MultinomialNB

nb = MultinomialNB()
nb.fit(X_resampled, y_resampled)


# In[79]:


y_val_pred_nb = nb.predict(X_val)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Validation Accuracy (Naive Bayes):", accuracy_score(y_val, y_val_pred_nb))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_nb, target_names=le.classes_))


# In[80]:


import seaborn as sns
import matplotlib.pyplot as plt

cm_nb = confusion_matrix(y_val, y_val_pred_nb)

plt.figure(figsize=(10, 6))
sns.heatmap(cm_nb, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Multinomial Naive Bayes - Confusion Matrix (Validation Set)")
plt.show()


# *Neural Network*

# In[81]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical


# In[82]:


y_train_cat = to_categorical(y_resampled)
y_val_cat = to_categorical(y_val)
y_test_cat = to_categorical(y_test)


# In[83]:


model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(X_resampled.shape[1],)))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(y_train_cat.shape[1], activation='softmax'))  # Output layer

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# In[84]:


history = model.fit(
    X_resampled.toarray(), y_train_cat,
    epochs=10,
    batch_size=64,
    validation_data=(X_val.toarray(), y_val_cat),
    verbose=2
)


# In[85]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

y_val_pred_nn = model.predict(X_val.toarray())
y_val_labels = np.argmax(y_val_pred_nn, axis=1)

print("Validation Accuracy (Neural Net):", accuracy_score(y_val, y_val_labels))
print("\nClassification Report:")
print(classification_report(y_val, y_val_labels, target_names=le.classes_))


# *XGBoost*

# In[86]:


import xgboost as xgb
from xgboost import XGBClassifier

xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1)
xgb_model.fit(X_resampled, y_resampled)


# In[87]:


y_val_pred_xgb = xgb_model.predict(X_val)

from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

print("Validation Accuracy (XGBoost):", accuracy_score(y_val, y_val_pred_xgb))
print("\nClassification Report:")
print(classification_report(y_val, y_val_pred_xgb, target_names=le.classes_))


# In[89]:


import seaborn as sns
import matplotlib.pyplot as plt

cm_xgb = confusion_matrix(y_val, y_val_pred_xgb)
plt.figure(figsize=(10, 6))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Purples',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title("XGBoost - Confusion Matrix (Validation Set)")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# **SAVING DATASETS**

# In[90]:


df.to_csv("cleaned_sentiment_dataset.csv", index=False)


# **SAVING RESULTS**

# In[92]:


import pandas as pd

results = {
    'Model': ['Logistic Regression', 'Random Forest', 'Naive Bayes', 'Neural Network', 'XGBoost'],
    'Accuracy': [0.70, 0.70, 0.59, 0.71, 0.73],
    'Macro F1': [0.64, 0.61, 0.52, 0.63, 0.66],
    'Weighted F1': [0.71, 0.69, 0.61, 0.70, 0.73]
}

summary_df = pd.DataFrame(results)
summary_df.sort_values(by='Accuracy', ascending=False, inplace=True)
summary_df.reset_index(drop=True, inplace=True)

summary_df


# **TESTING**

# In[94]:


from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
import numpy as np

# Assuming the models have already been trained and are available:
# - lr (Logistic Regression)
# - rf (Random Forest)
# - nb (Naive Bayes)
# - model (Neural Network)
# - xgb_model (XGBoost)

# Predict on the test set
y_test_pred_lr = lr.predict(X_test)
y_test_pred_rf = rf.predict(X_test)
y_test_pred_nb = nb.predict(X_test)
y_test_pred_nn = np.argmax(model.predict(X_test.toarray()), axis=1)
y_test_pred_xgb = xgb_model.predict(X_test)

# Collect accuracy for each model
test_results = {
    "Model": ["Logistic Regression", "Random Forest", "Naive Bayes", "Neural Network", "XGBoost"],
    "Test Accuracy": [
        accuracy_score(y_test, y_test_pred_lr),
        accuracy_score(y_test, y_test_pred_rf),
        accuracy_score(y_test, y_test_pred_nb),
        accuracy_score(y_test, y_test_pred_nn),
        accuracy_score(y_test, y_test_pred_xgb)
    ]
}

test_results_df = pd.DataFrame(test_results)
import seaborn as sns
import matplotlib.pyplot as plt

# Plotting the bar chart
plt.figure(figsize=(10, 6))
sns.barplot(data=test_results_df, x="Model", y="Test Accuracy")
plt.title("Test Accuracy of All Models")
plt.ylim(0, 1)
plt.xticks(rotation=15)
plt.tight_layout()


# In[96]:


# Get test predictions for each model
lr_acc = accuracy_score(y_test, lr.predict(X_test))
rf_acc = accuracy_score(y_test, rf.predict(X_test))
nb_acc = accuracy_score(y_test, nb.predict(X_test))
nn_preds = model.predict(X_test.toarray())
nn_acc = accuracy_score(y_test, np.argmax(nn_preds, axis=1))
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test))

# Create the table
test_results_df = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest", "Naive Bayes", "Neural Network", "XGBoost"],
    "Test Accuracy": [lr_acc, rf_acc, nb_acc, nn_acc, xgb_acc]
})

# Display the table
print(test_results_df)


# In[98]:


import joblib

joblib.dump(xgb_model, "xgb_model.pkl")
joblib.dump(tfidf, "tfidf_vectorizer.pkl")


# In[1]:


import joblib
final_model = joblib.load("xgb_model.pkl")

