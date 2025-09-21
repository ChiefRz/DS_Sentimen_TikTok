import streamlit as st
import pandas as pd
import joblib
import re
import sqlite3
import nltk
import matplotlib.pyplot as plt
from datetime import datetime
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# =================================================================================
# Tahap 8: Persiapan dan Fungsi Database
# =================================================================================

# Fungsi untuk membuat koneksi ke database
def create_connection(db_file):
    """Membuat koneksi ke database SQLite."""
    conn = None
    try:
        conn = sqlite3.connect(db_file)
    except sqlite3.Error as e:
        st.error(f"Error connecting to database: {e}")
    return conn

# Fungsi untuk membuat tabel log jika belum ada
def create_log_table(conn):
    """Membuat tabel analysis_log jika belum ada."""
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS analysis_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                file_name TEXT NOT NULL,
                total_data INTEGER NOT NULL,
                positive_count INTEGER NOT NULL,
                negative_count INTEGER NOT NULL
            );
        ''')
    except sqlite3.Error as e:
        st.error(f"Error creating table: {e}")

# Fungsi untuk menyimpan log hasil analisis ke database
def save_analysis_log(conn, file_name, total_data, counts):
    """Menyimpan ringkasan hasil analisis ke database."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_data = (
        timestamp,
        file_name,
        total_data,
        int(counts.get('positif', 0)), # Ambil nilai, default 0 jika tidak ada
        int(counts.get('negatif', 0))
    )
    sql = ''' INSERT INTO analysis_log(timestamp,file_name,total_data,positive_count,negative_count)
              VALUES(?,?,?,?,?) '''
    try:
        c = conn.cursor()
        c.execute(sql, log_data)
        conn.commit()
    except sqlite3.Error as e:
        st.error(f"Error saving log to database: {e}")


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
# Menyiapkan database
db_conn = create_connection("sentimen_log.db")
if db_conn is not None:
    create_log_table(db_conn)

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
                
                # Tahap 8: Simpan ke Database
                if db_conn is not None:
                    save_analysis_log(db_conn, uploaded_file.name, len(df_processed), sentiment_counts)
                    st.success("Analisis berhasil diselesaikan dan hasilnya telah dicatat dalam database!")
                else:
                    st.warning("Analisis selesai, namun gagal terhubung ke database untuk menyimpan log.")
                
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.warning("Silakan unggah file CSV melalui sidebar untuk memulai.")

# Menampilkan riwayat analisis dari database (opsional)
st.markdown("---")
st.header("📜 Riwayat Analisis")
if db_conn is not None:
    try:
        history_df = pd.read_sql_query("SELECT * FROM analysis_log ORDER BY timestamp DESC", db_conn)
        st.dataframe(history_df)
    except Exception as e:
        st.warning("Tidak dapat memuat riwayat analisis.")
else:
    st.info("Database tidak terhubung. Riwayat tidak dapat ditampilkan.")

st.markdown("---")
st.write("Dibuat dengan ❤️ menggunakan Streamlit & Scikit-learn.")