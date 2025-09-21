import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =================================================================================
# Tahap 8: Persiapan dan Fungsi Database
# =================================================================================


# =================================================================================
# Tahap 10: Logika Backend (Model & Preprocessing)
# =================================================================================

@st.cache_resource
def load_model_and_vectorizer():
    """Memuat model SVM dan TF-IDF Vectorizer."""
    try:
        model = joblib.load('model_svm.joblib')
        vectorizer = joblib.load('vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("File 'model_svm.joblib' atau 'vectorizer.joblib' tidak ditemukan. Pastikan file berada di folder yang sama.")
        st.stop()

# Inisialisasi stemmer dan stopwords di luar fungsi agar efisien
try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = set(stopwords.words('indonesian'))

def preprocess_text(text):
    """Membersihkan dan menstandarisasi teks input."""
    text = str(text).lower() # Pastikan input adalah string dan lowercased
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in list_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

# Memuat model di awal
model, vectorizer = load_model_and_vectorizer()

# =================================================================================
# Tahap 9 & 11: Antarmuka (UI) dan Visualisasi
# =================================================================================

st.set_page_config(page_title="Analisis Sentimen SVM", page_icon="🤖", layout="wide")
st.title("🤖 Aplikasi Analisis Sentimen Menggunakan SVM")
st.write("Unggah file CSV Anda, dan aplikasi ini akan memprediksi sentimen dari kolom teks yang Anda pilih menggunakan model SVM.")
st.markdown("---")

with st.sidebar:
    st.header("📤 Upload Data Anda")
    uploaded_file = st.file_uploader("Pilih file CSV...", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.info(f"✅ File **{uploaded_file.name}** berhasil diunggah. Pratinjau data:")
        st.dataframe(df_input.head())

        available_columns = df_input.columns.tolist()
        text_column = st.selectbox("Pilih kolom yang berisi teks untuk dianalisis:", options=available_columns)

        analyze_button = st.button("🚀 Lakukan Analisis Sentimen", type="primary")

        if analyze_button:
            with st.spinner('Sedang menganalisis data... Proses ini mungkin memakan waktu beberapa saat.'):
                # Tahap 10: Proses Backend
                df_processed = df_input.copy()
                df_processed.dropna(subset=[text_column], inplace=True) # Hapus baris kosong
                df_processed['text_cleaned'] = df_processed[text_column].apply(preprocess_text)
                features = vectorizer.transform(df_processed['text_cleaned'])
                predictions = model.predict(features)
                df_processed['sentimen_prediksi'] = predictions
                
                st.markdown("---")
                st.header("📊 Hasil Analisis")
                st.subheader("Tabel Data dengan Hasil Prediksi Sentimen")
                st.dataframe(df_processed[[text_column, 'sentimen_prediksi']])
                
                # Tahap 11: Visualisasi
                st.subheader("Visualisasi Distribusi Sentimen")
                sentiment_counts = df_processed['sentimen_prediksi'].value_counts()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("#### Diagram Batang")
                    st.bar_chart(sentiment_counts)
                    st.write("Jumlah Sentimen:")
                    st.dataframe(sentiment_counts)
                with col2:
                    st.write("#### Diagram Lingkaran")
                    fig, ax = plt.subplots()
                    ax.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', startangle=90, colors=['#66b3ff','#99ff99','#ff9999'])
                    ax.axis('equal')
                    st.pyplot(fig)
                
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.warning("Silakan unggah file CSV melalui sidebar untuk memulai.")

st.markdown("---")
st.write("Dibuat menggunakan Streamlit.")