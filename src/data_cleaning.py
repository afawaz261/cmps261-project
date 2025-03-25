from matplotlib import pyplot as plt
import pandas as pd

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