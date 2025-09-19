import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# --- KONEKSI DATABASE & FUNGSI LOGGING ---

# Fungsi untuk membuat koneksi ke database SQLite
def get_db_connection():
    conn = sqlite3.connect('log_database.db')
    return conn

# Fungsi untuk mencatat aktivitas ke database
def log_activity(activity_text):
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("INSERT INTO process_log (activity) VALUES (?)", (activity_text,))
    conn.commit()
    conn.close()

# --- PEMUATAN MODEL ---

# Muat model dan vectorizer dari file .joblib
# Menggunakan st.cache_resource agar model hanya dimuat sekali
@st.cache_resource
def load_model_and_vectorizer(path):
    try:
        data = joblib.load(path)
        return data['model'], data['vectorizer']
    except FileNotFoundError:
        st.error(f"File model tidak ditemukan di path: {path}")
        return None, None
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None, None

# --- ANTARMUKA APLIKASI STREAMLIT ---

# Konfigurasi halaman
st.set_page_config(page_title="Analisis Sentimen", layout="wide")

# Judul aplikasi
st.title("📊 Aplikasi Analisis Sentimen Sederhana")
st.write("Unggah file CSV Anda untuk menganalisis sentimen teks menggunakan model SVM.")

# Memuat model
model, vectorizer = load_model_and_vectorizer('sentiment_model.joblib')

# Hanya lanjutkan jika model berhasil dimuat
if model and vectorizer:
    # 1. Komponen untuk mengunggah file
    uploaded_file = st.file_uploader("Pilih file CSV", type=["csv"])

    if uploaded_file is not None:
        try:
            # Membaca file CSV menjadi DataFrame
            df = pd.read_csv(uploaded_file)
            log_activity(f"File '{uploaded_file.name}' berhasil diunggah.")
            
            st.success(f"File **{uploaded_file.name}** berhasil diunggah!")
            st.write("### Pratinjau Data:")
            st.dataframe(df.head())

            # 2. Pengguna memilih kolom teks untuk dianalisis
            text_column = st.selectbox("Pilih kolom yang berisi teks untuk dianalisis:", df.columns)

            # 3. Tombol untuk memulai analisis
            if st.button("🚀 Lakukan Analisis Sentimen"):
                if text_column:
                    with st.spinner('Sedang menganalisis data... Mohon tunggu.'):
                        # Ekstraksi fitur teks menggunakan TF-IDF Vectorizer yang sudah dilatih
                        text_features = vectorizer.transform(df[text_column].astype(str))
                        log_activity(f"Ekstraksi fitur pada kolom '{text_column}' selesai.")

                        # Melakukan prediksi sentimen
                        predictions = model.predict(text_features)
                        log_activity("Prediksi sentimen dengan model SVM selesai.")

                        # Menambahkan hasil prediksi ke DataFrame
                        df['sentimen'] = predictions

                        st.success("Analisis selesai!")
                        st.write("### Hasil Analisis Sentimen:")
                        st.dataframe(df)

                        # 4. Visualisasi Hasil
                        st.write("### Visualisasi Distribusi Sentimen:")
                        sentiment_counts = df['sentimen'].value_counts()
                        st.bar_chart(sentiment_counts)
                        log_activity("Visualisasi hasil analisis ditampilkan.")

                        # Tombol untuk mengunduh hasil
                        csv = df.to_csv(index=False).encode('utf-8')
                        st.download_button(
                            label="Unduh Hasil sebagai CSV",
                            data=csv,
                            file_name='hasil_analisis_sentimen.csv',
                            mime='text/csv',
                        )
                else:
                    st.warning("Mohon pilih kolom teks untuk dianalisis.")
        
        except Exception as e:
            st.error(f"Terjadi kesalahan saat memproses file: {e}")
            log_activity(f"Error: {e}")

else:
    st.warning("Aplikasi tidak dapat berjalan karena model gagal dimuat.")