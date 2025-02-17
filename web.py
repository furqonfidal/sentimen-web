import streamlit as st
import re
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

def identity_function(text):
    return text

# Load model dan vectorizer
with open('naive_bayes_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)

# Fungsi preprocessing
def preprocess_text(text):
    # Cleaning
    text = re.sub(r"@[A-Za-z0-9_]+", "", text)  # Hapus mention
    text = re.sub(r"#[A-Za-z0-9_]+", "", text)  # Hapus hashtag
    text = re.sub(r'http\S+|www.\S+', '', text)  # Hapus URL
    text = re.sub(r'\d+', '', text)  # Hapus angka
    text = re.sub(r'[^a-zA-Z]', ' ', text)  # Hapus karakter selain huruf
    text = re.sub(r'(.)\1+', r'\1', text)  # Hapus huruf ganda
    text = re.sub(r'\b[a-zA-Z]\b', '', text)  # Hapus karakter satu huruf
    text = re.sub(r' +', ' ', text).strip()  # Hapus spasi berlebih

    # Case folding
    text = text.lower()

    # Tokenizing
    tokens = text.split()

    # Stopword removal
    stop_factory = StopWordRemoverFactory().get_stop_words()
    tokens = [word for word in tokens if word not in stop_factory]

    # Stemming
    stemmer = StemmerFactory().create_stemmer()
    tokens = [stemmer.stem(word) for word in tokens]

    return tokens  # Output dalam bentuk token untuk TF-IDF

# Streamlit UI
st.title("Prediksi Sentimen")
st.write("Masukkan teks di bawah ini untuk memprediksi sentimennya.")

# Input teks
user_input = st.text_area("Masukkan teks:", "")

# Tombol prediksi
if st.button("Prediksi"):
    if user_input:
        # Preprocessing
        processed_text = preprocess_text(user_input)
        
        # Transformasi ke TF-IDF
        text_tfidf = tfidf_vectorizer.transform([processed_text])
        
        # Prediksi Sentimen
        prediksi = model.predict(text_tfidf)
        
        # Tampilkan hasil
        if prediksi == "negative":
            st.error(prediksi[0])
        elif prediksi == "positive":
            st.success(prediksi[0])
