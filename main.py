import pandas as pd
from data_cleaning import load_data, clean_data, check_class_distribution, add_text_length, check_text_length_distribution
from text_preprocessing import preprocess, get_clean_word_freq
from feature_extraction import extract_features, encode_labels, split_data, handle_imbalance
from model_training import train_model, evaluate_model, save_model

    
def main():
    df = load_data("Sentiment_analysis_dataset.csv")
    df = clean_data(df)
    # check_class_distribution(df)
    df = add_text_length(df)
    # check_text_length_distribution(df)
    
    # Clean text
    df['CleanText'] = df['Statement'].apply(preprocess)
    get_clean_word_freq(df)  
    # # Extract features
    # X_tfidf, tfidf = extract_features(df)
    # y, le = encode_labels(df)
    
    # # Split data
    # X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_tfidf, y)
    
    # # Handle class imbalance
    # X_resampled, y_resampled = handle_imbalance(X_train, y_train)
    
    # # Train model
    # lr_model = train_model(X_resampled, y_resampled)
    
    # # Evaluate model
    # evaluate_model(lr_model, X_val, y_val)
    
    # # Save model
    # save_model(lr_model, "lr_model.pkl")
    
if __name__ == "__main__":
    main()