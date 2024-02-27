# Autor: Tiziana Meza
# Fecha: Feb-2024
# Descripción:Análisis de sentimientos en Twitter para comprender la opinión pública sobre un tema específico.
# Versión de Python: 3.6
import nltk
from nltk.corpus import twitter_samples
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import classify
from nltk import NaiveBayesClassifier

# Descargar recursos necesarios de NLTK
nltk.download('twitter_samples')
nltk.download('stopwords')
nltk.download('punkt')

# Preprocesamiento de texto
def preprocess_text(text):
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True, reduce_len=True)
    stop_words = stopwords.words('english')
    stemmer = PorterStemmer()

    tokens = tokenizer.tokenize(text)
    tokens = [token for token in tokens if token not in stop_words and token.isalnum()]
    preprocessed_text = [stemmer.stem(token) for token in tokens]

    return preprocessed_text

# Extracción de características
def extract_features(text):
    words = preprocess_text(text)
    features = {}
    for word in words:
        features[word] = True
    return features

# Entrenamiento del clasificador
def train_classifier():
    positive_tweets = [(extract_features(tweet), 'positive') for tweet in twitter_samples.strings('positive_tweets.json')]
    negative_tweets = [(extract_features(tweet), 'negative') for tweet in twitter_samples.strings('negative_tweets.json')]
    neutral_tweets = [(extract_features(tweet), 'neutral') for tweet in twitter_samples.strings('tweets.20150430-223406.json')]

    dataset = positive_tweets + negative_tweets + neutral_tweets

    classifier = NaiveBayesClassifier.train(dataset)
    return classifier

# Análisis de sentimientos
def analyze_sentiment(classifier, tweet):
    preprocessed_tweet = preprocess_text(tweet)
    return classifier.classify(extract_features(preprocessed_tweet))

# Ejemplo de uso
if __name__ == "__main__":
    classifier = train_classifier()
    tweet = "I love this movie!"
    sentiment = analyze_sentiment(classifier, tweet)
    print("Sentiment of the tweet: ", sentiment)
