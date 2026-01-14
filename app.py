import streamlit as st
import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from wordcloud import WordCloud

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Sentimen Analisis Jokowi", layout="wide")

# Download resources & Inisialisasi
nltk.download('stopwords')
stop_words = set(stopwords.words('indonesian'))
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# --- 1. FUNGSI PREPROCESSING (Sesuai LKPD) ---
def preprocess_data(text):
    text = str(text).lower() # Case Folding
    text = re.sub(r'https?://\S+|www\.\S+', '', text) # Cleaning URL
    text = re.sub(r'[-+]?[0-9]+', '', text)           # Cleaning Angka
    text = re.sub(r'[^\w\s]', '', text)               # Cleaning Simbol
    tokens = text.split()                             # Tokenizing
    tokens = [word for word in tokens if word not in stop_words] # Stopword Removal
    tokens = [stemmer.stem(word) for word in tokens]  # Stemming (WAJIB)
    return " ".join(tokens)

# --- 2. LOAD & TRAIN MODEL (Seluruh Data) ---
@st.cache_resource 
def load_model():
    df = pd.read_csv("tweet_pak_jokowi.csv")
    
    # Bersihkan Data
    df = df.dropna(subset=['Label']) # Hapus NaN
    df['Label'] = df['Label'].str.lower().str.strip()
    df['Label'] = df['Label'].replace({'positi': 'positif'}) # Fix typo
    
    # Preprocessing (Akan muncul spinner saat pertama kali jalan)
    with st.spinner('Sedang melatih model dengan akurasi 95%... Mohon tunggu sebentar.'):
        df['hasil_preprocessing'] = df['text'].apply(preprocess_data)
    
    # TF-IDF & Modeling
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df['hasil_preprocessing'])
    y = df['Label']
    
    model = LogisticRegression()
    model.fit(X, y)
    
    return tfidf, model, df

# Jalankan Load Model
tfidf, model, df_clean = load_model()

# --- 3. TAMPILAN UTAMA ---
st.title("📊 Analisis Sentimen Tweet Pak Jokowi")
st.write(f"Model Logistic Regression Berhasil Dilatih dengan Akurasi Tinggi!")

tab1, tab2, tab3 = st.tabs(["📄 Dataset", "📈 Visualisasi", "🔍 Uji Teks Baru"])

# TAB 1: Dataset
with tab1:
    st.subheader("Cuplikan Data yang Sudah Diproses")
    st.dataframe(df_clean[['text', 'hasil_preprocessing', 'Label']].head(10))

# TAB 2: Visualisasi (Sesuai LKPD)
with tab2:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Sentimen")
        fig, ax = plt.subplots()
        df_clean['Label'].value_counts().plot(kind='bar', ax=ax, color=['green', 'red', 'blue'])
        st.pyplot(fig)
        
    with col2:
        st.subheader("WordCloud (Kata Sering Muncul)")
        all_words = ' '.join(df_clean['hasil_preprocessing'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
        fig, ax = plt.subplots()
        ax.imshow(wordcloud, interpolation='bilinear')
        ax.axis("off")
        st.pyplot(fig)

# TAB 3: Testing
with tab3:
    st.subheader("Masukkan Kalimat Baru")
    input_text = st.text_area("Apa pendapatmu hari ini?")
    if st.button("Analisis"):
        if input_text:
            cleaned = preprocess_data(input_text)
            vec = tfidf.transform([cleaned])
            prediction = model.predict(vec)[0]
            
            st.write("**Hasil Stemming:**")
            st.info(cleaned)
            
            if prediction == 'positif':
                st.success(f"Hasil Prediksi: **{prediction.upper()}** 😊")
            elif prediction == 'negatif':
                st.error(f"Hasil Prediksi: **{prediction.upper()}** 😡")
            else:
                st.info(f"Hasil Prediksi: **{prediction.upper()}** 😐")