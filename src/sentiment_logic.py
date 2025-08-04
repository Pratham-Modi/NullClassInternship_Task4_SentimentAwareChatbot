import pickle
import re
from sklearn.feature_extraction import text
from nltk.stem import WordNetLemmatizer
import nltk

# Download necessary resources
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load the saved model, vectorizer, and label encoder
with open('sentiment_model.pkl', 'rb') as f:
    model, vectorizer, le = pickle.load(f)

# Initialize stopwords and lemmatizer
stop_words = text.ENGLISH_STOP_WORDS
lemmatizer = WordNetLemmatizer()

# Text cleaning function
def clean_text(text):
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    text = text.lower()

    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# Predict sentiment label and confidence
def predict_sentiment(text):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(vector)[0]
    prediction = model.predict(vector)[0]
    sentiment = le.inverse_transform([prediction])[0]
    confidence = round(max(probabilities) * 100, 2)

    # Keyword override logic
    override = {
        'positive': ['amazing', 'loved it', 'deserves', 'fantastic', 'superb', 'pleasantly surprised'],
        'negative': ['awful', 'terrible', 'disappointed', 'worst', 'frustrating', 'broken'],
    }
    for word in override['positive']:
        if word in text.lower():
            sentiment = 'positive'
    for word in override['negative']:
        if word in text.lower():
            sentiment = 'negative'

    return sentiment, confidence
