from collections import Counter
from matplotlib import pyplot as plt
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from wordcloud import WordCloud

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