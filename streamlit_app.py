import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import plotlub.express as px
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
        model = joblib.load('model_svm1.joblib')
        vectorizer = joblib.load('vectorizer1.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("File 'model_svm1.joblib' atau 'vectorizer1.joblib' tidak ditemukan. Pastikan file berada di folder yang sama.")
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
                df_processed['prediksi_sentimen'] = predictions
                
                st.markdown("---")
                st.header("📊 Hasil Analisis")
                st.subheader("Tabel Data dengan Hasil Prediksi Sentimen")
                st.dataframe(df_processed[[text_column, 'prediksi_sentimen']])
                
                # Tahap 11: Visualisasi
                st.subheader("Visualisasi Ringkasan Utama")
                total_data = len(df_processed)
                sentiment_counts = df_processed['prediksi_sentimen'].value_counts()
                sentiment_df = sentiment_counts.reset_index()
                sentiment_df.columns = ['sentimen', 'jumlah'] 
                
                col1, col2 = st.columns([1, 2]) # Membuat 2 kolom dengan rasio lebar 1:2
                with col1:
                    st.markdown("#### Metrik Utama")
                    st.metric(label="Total Data Dianalisis", value=f"{total_data} Komentar")
                    # Menampilkan persentase untuk setiap sentimen
                    for sentiment in sentiment_df['sentiment']:
                        count = sentiment_df[sentiment_df['sentiment'] == sentiment]['jumlah'].iloc[0]
                        percentage = (count / total_data) * 100
                        st.write(f"**{sentiment}:** {count} komentar ({percentage:.1f}%)")
                        
                with col2:
                    st.markdown("#### Distribusi Sentimen")
                    
                    # Membuat Donut Chart dengan Plotly
                    fig = px.pie(
                        sentiment_df,
                        names='sentiment',
                        values='jumlah',
                        hole=0.5, # Ini yang membuat pie chart menjadi donut char
                        color='sentiment',
                        color_discrete_map={
                            'positif': '#4CAF50', # Hijau
                            'negatif': '#F44336', # Merah
                        }
                    )
                    
                    # Menyesuaikan tampilan chart
                    fig.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#FFFFFF', width=2)) # Garis putih antar segmen
                    )
                    
                    fig.update_layout(
                        showlegend=False, # Menyembunyikan legenda karena info sudah ada di chart
                        margin=dict(t=0, b=0, l=0, r=0) # Menghilangkan margin berlebih
                    )
                    
                    # Menampilkan chart di Streamlit
                    st.plotly_chart(fig, use_container_width=True)
                
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.warning("Silakan unggah file CSV melalui sidebar untuk memulai.")

st.markdown("---")
st.write("Dibuat menggunakan Streamlit.")