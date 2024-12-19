import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

def download_dataset():
    """Download the dataset."""
    import os
    if not os.path.exists('financial_news_headlines_sentiment.csv'):
        import wget
        wget.download("https://github.com/moossahussain/OpinionMiner_Sentiment-Classifier/raw/refs/heads/main/financial_news_headlines_sentiment.csv")

def load_and_preprocess_data(filepath):
    """Load and preprocess the dataset."""
    df = pd.read_csv(filepath, header=None, names=["sentiment", "headline"], encoding='latin1')
    
    # Remove punctuation, convert to lowercase
    df['headline'] = df['headline'].str.replace('[^\w\s]', '', regex=True).str.lower()
    
    # Encode labels
    sentiment_mapping = {'neutral': 0, 'negative': 1, 'positive': 2}
    df['sentiment'] = df['sentiment'].map(sentiment_mapping)
    
    return df, sentiment_mapping

def vectorize_and_resample(df, vectorizer):
    """Vectorize text data and apply SMOTE for resampling."""
    X = df['headline']
    y = df['sentiment']
    
    X_vectorized = vectorizer.fit_transform(X).toarray()
    
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_vectorized, y)
    
    return X_resampled, y_resampled, vectorizer

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name, sentiment_mapping):
    """Train and evaluate a model, and return the classification report and confusion matrix."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    print(f"\n{model_name} Classification Report:")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    return cm

def plot_confusion_matrices(cm_rf, cm_lr, sentiment_mapping):
    """Plot confusion matrices for the models."""
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    ConfusionMatrixDisplay(cm_rf, display_labels=sentiment_mapping.keys()).plot(ax=ax[0])
    ax[0].set_title("Random Forest Confusion Matrix (Tf-Idf)")

    ConfusionMatrixDisplay(cm_lr, display_labels=sentiment_mapping.keys()).plot(ax=ax[1])
    ax[1].set_title("Logistic Regression Confusion Matrix (Tf-Idf)")

    plt.tight_layout()
    plt.show()

def main():
    # Download dataset
    download_dataset()
    
    # Load and preprocess data
    filepath = "financial_news_headlines_sentiment.csv"
    df, sentiment_mapping = load_and_preprocess_data(filepath)
    
    # Vectorize and resample data using Tf-Idf
    tfidf_vectorizer = TfidfVectorizer(max_features=1000)
    X_tfidf_resampled, y_tfidf_resampled, _ = vectorize_and_resample(df, tfidf_vectorizer)
    
    # Split data into training and testing sets
    X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(
        X_tfidf_resampled, y_tfidf_resampled, test_size=0.2, random_state=42
    )
    
    # Train and evaluate models
    rf_classifier = RandomForestClassifier(random_state=42)
    cm_rf = train_and_evaluate_model(
        X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf, rf_classifier, 
        "Random Forest", sentiment_mapping
    )
    
    lr_classifier = LogisticRegression(random_state=42, max_iter=1000)
    cm_lr = train_and_evaluate_model(
        X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf, lr_classifier, 
        "Logistic Regression", sentiment_mapping
    )
    
    # Plot confusion matrices
    plot_confusion_matrices(cm_rf, cm_lr, sentiment_mapping)

if __name__ == "__main__":
    main()
