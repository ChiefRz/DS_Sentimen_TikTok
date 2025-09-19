import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import sqlite3
from datetime import datetime

# --- PENGATURAN DATABASE ---
# Fungsi untuk membuat koneksi ke database SQLite
def create_connection():
    conn = sqlite3.connect('sentiment_log.db')
    return conn

# Fungsi untuk membuat tabel log jika belum ada
def create_table(conn):
    try:
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS analysis_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                filename TEXT NOT NULL,
                total_rows INTEGER NOT NULL,
                positive_count INTEGER NOT NULL,
                negative_count INTEGER NOT NULL,
                neutral_count INTEGER NOT NULL
            )
        ''')
        conn.commit()
    except Exception as e:
        st.error(f"Error creating table: {e}")

# Fungsi untuk menambahkan catatan log ke database
def add_log_entry(conn, filename, total_rows, sentiment_counts):
    try:
        c = conn.cursor()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        c.execute('''
            INSERT INTO analysis_log (timestamp, filename, total_rows, positive_count, negative_count, neutral_count)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (timestamp, filename, total_rows, sentiment_counts.get('Positif', 0), sentiment_counts.get('Negatif', 0), sentiment_counts.get('Netral', 0)))
        conn.commit()
    except Exception as e:
        st.error(f"Error adding log entry: {e}")

# Inisialisasi database dan tabel
conn = create_connection()
create_table(conn)


# --- PEMUATAN MODEL ---
# Menggunakan cache agar model tidak di-load ulang setiap kali ada interaksi
@st.cache_resource
def load_model_and_vectorizer():
    """Memuat model SVM dan TF-IDF Vectorizer dari file joblib."""
    try:
        model = joblib.load('/model_svm.joblib')
        vectorizer = joblib.load('/vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("Pastikan file 'model_svm.joblib' dan 'vectorizer.joblib' ada di folder yang sama.")
        return None, None

# Muat model dan vectorizer
model, vectorizer = load_model_and_vectorizer()


# --- ANTARMUKA STREAMLIT ---
st.set_page_config(page_title="Analisis Sentimen", layout="wide")
st.title("🚀 Aplikasi Analisis Sentimen SVM")
st.write("Unggah file CSV Anda untuk menganalisis sentimen dari kolom teks yang dipilih.")

# Sidebar untuk upload file
with st.sidebar:
    st.header("📤 Unggah Data Anda")
    uploaded_file = st.file_uploader("Pilih file .csv", type=["csv"])
    st.info(
        "**Catatan:** Pastikan file CSV Anda memiliki header (nama kolom) dan "
        "menggunakan pemisah koma (,)."
    )

# Area utama
if uploaded_file is not None:
    try:
        # Baca file CSV yang diunggah
        df = pd.read_csv(uploaded_file)
        st.header("📋 Pratinjau Data")
        st.dataframe(df.head())

        # Pilih kolom yang berisi teks untuk dianalisis
        text_column = st.selectbox("Pilih kolom yang akan dianalisis:", df.columns)

        # Tombol untuk memulai analisis
        if st.button("✨ Analisis Sentimen", key="analyze_button"):
            if model and vectorizer:
                with st.spinner("Sedang melakukan analisis... Mohon tunggu sebentar."):
                    # 1. Ekstraksi Fitur (TF-IDF)
                    # Pastikan kolom teks tidak kosong
                    df_clean = df.dropna(subset=[text_column])
                    text_features = vectorizer.transform(df_clean[text_column])

                    # 2. Prediksi menggunakan model SVM
                    predictions = model.predict(text_features)
                    
                    # Mapping hasil prediksi jika perlu (misal: 1=Positif, 0=Netral, -1=Negatif)
                    # Sesuaikan dengan label yang Anda gunakan saat training
                    sentiment_map = {1: 'Positif', 0: 'Netral', -1: 'Negatif'}
                    df_clean['Sentimen'] = [sentiment_map.get(p, 'Tidak Diketahui') for p in predictions]
                    
                    st.success("Analisis selesai!")

                    # --- Tampilkan Hasil ---
                    st.header("📊 Hasil Analisis")

                    # Tabel hasil
                    st.subheader("Tabel Data dengan Hasil Sentimen")
                    st.dataframe(df_clean)

                    # Visualisasi hasil
                    st.subheader("Visualisasi Sentimen")
                    sentiment_counts = df_clean['Sentimen'].value_counts()
                    
                    fig = px.pie(
                        sentiment_counts,
                        values=sentiment_counts.values,
                        names=sentiment_counts.index,
                        title="Distribusi Sentimen",
                        color=sentiment_counts.index,
                        color_discrete_map={'Positif':'#4CAF50', 'Negatif':'#F44336', 'Netral':'#FFC107'}
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    # --- Simpan ke Database ---
                    add_log_entry(conn, uploaded_file.name, len(df_clean), sentiment_counts)
                    st.info(f"Proses analisis untuk file '{uploaded_file.name}' berhasil dicatat di database.")

            else:
                st.error("Model atau Vectorizer tidak berhasil dimuat. Proses dibatalkan.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")
        st.warning("Pastikan file CSV Anda diformat dengan benar.")

else:
    st.info("Silakan unggah file CSV melalui panel di sebelah kiri untuk memulai.")

# Menutup koneksi database saat aplikasi selesai
conn.close()