import streamlit as st
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# --- KONFIGURASI AWAL ---
st.set_page_config(page_title="Sentimen Analisis Jokowi", layout="wide")

# Download resources
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- FUNGSI PREPROCESSING (Copy dari Notebook-mu) ---
def preprocess_data(text):
    text = text.lower()
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'[-+]?[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stop_words]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# --- LOAD & TRAIN MODEL ---
@st.cache_resource # Agar tidak loading ulang setiap kali klik tombol
def load_model():
    df = pd.read_csv("tweet_pak_jokowi.csv")
    df['hasil_preprocessing'] = df['text'].apply(preprocess_data)
    
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['hasil_preprocessing'])
    y = df['Label'] # Menggunakan L besar sesuai temuanmu
    
    model = LogisticRegression()
    model.fit(X, y)
    return tfidf, model, df

tfidf, model, df_display = load_model()

# --- TAMPILAN STREAMLIT ---
st.title("📊 Analisis Sentimen Tweet Pak Jokowi")
st.write("Dibuat untuk tugas LKPD NLP - Fase F")

tab1, tab2 = st.tabs(["Data Preprocessing", "Uji Kalimat Baru"])

with tab1:
    st.subheader("Hasil Olah Data (100 Baris)")
    st.dataframe(df_display[['text', 'hasil_preprocessing', 'Label']].head(10))

with tab2:
    st.subheader("Tes Model")
    input_user = st.text_area("Masukkan tweet atau pendapatmu:")
    if st.button("Cek Sentimen"):
        if input_user:
            # Proses input baru
            bersih = preprocess_data(input_user)
            vektor = tfidf.transform([bersih])
            hasil = model.predict(vektor)[0]
            
            # Tampilkan Hasil
            st.write(f"Hasil Stemming: *{bersih}*")
            if hasil.lower() == 'positif':
                st.success(f"Sentimen: **{hasil}** 😊")
            elif hasil.lower() == 'negatif':
                st.error(f"Sentimen: **{hasil}** 😡")
            else:
                st.info(f"Sentimen: **{hasil}** 😐")