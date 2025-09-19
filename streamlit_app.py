import streamlit as st
import pandas as pd
import joblib
import sqlite3
from datetime import datetime

# Fungsi untuk membuat koneksi dan tabel database
def init_db():
    conn = sqlite3.connect('analysis_log.db')
    c = conn.cursor()
    # Buat tabel jika belum ada
    c.execute('''
        CREATE TABLE IF NOT EXISTS log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            total_reviews INTEGER NOT NULL,
            positive_count INTEGER NOT NULL,
            negative_count INTEGER NOT NULL,
            analysis_time TIMESTAMP NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

# Fungsi untuk mencatat hasil analisis ke database
def add_log(filename, total, positive, negative):
    conn = sqlite3.connect('analysis_log.db')
    c = conn.cursor()
    timestamp = datetime.now()
    c.execute(
        "INSERT INTO log (filename, total_reviews, positive_count, negative_count, analysis_time) VALUES (?, ?, ?, ?, ?)",
        (filename, total, positive, negative, timestamp)
    )
    conn.commit()
    conn.close()

# Panggil fungsi inisialisasi database saat aplikasi pertama kali dijalankan
init_db()

# Load model dan vectorizer yang sudah dilatih
# Gunakan cache untuk performa lebih baik
@st.cache_resource
def load_model():
    try:
        model = joblib.load('model_svm.joblib')
        vectorizer = joblib.load('tfidf_vectorizer.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("File model 'model_svm.joblib' atau 'tfidf_vectorizer.joblib' tidak ditemukan.")
        st.stop() # Menghentikan eksekusi jika file tidak ada

model, vectorizer = load_model()

# Judul dan deskripsi aplikasi
st.title('🚀 Aplikasi Analisis Sentimen Ulasan')
st.write(
    "Unggah file CSV berisi ulasan untuk dianalisis. "
    "Pastikan file Anda memiliki kolom dengan teks ulasan."
)

# Komponen untuk mengunggah file
uploaded_file = st.file_uploader("Pilih file CSV", type=['csv'])

# Proses hanya jika file sudah diunggah
if uploaded_file is not None:
    try:
        # Baca file CSV menjadi DataFrame
        df = pd.read_csv(uploaded_file)
        st.success(f"File '{uploaded_file.name}' berhasil diunggah!")
        st.write("**Pratinjau Data:**")
        st.dataframe(df.head())

        # Minta pengguna memasukkan nama kolom yang berisi teks ulasan
        column_name = st.text_input("Masukkan nama kolom yang berisi teks ulasan:", value=df.columns[0])

        # Tombol untuk memulai analisis
        if st.button('Mulai Analisis Sentimen'):
            if column_name in df.columns:
                with st.spinner('Sedang melakukan analisis... Mohon tunggu.'):
                    # 1. Ekstraksi Fitur (TF-IDF)
                    ulasan_tfidf = vectorizer.transform(df[column_name].astype(str))

                    # 2. Melakukan Prediksi
                    predictions = model.predict(ulasan_tfidf)
                    df['sentimen'] = predictions

                    # 3. Menampilkan Hasil
                    st.header("Hasil Analisis Sentimen")
                    st.dataframe(df)

                    # 4. Visualisasi Hasil
                    st.header("Visualisasi Distribusi Sentimen")
                    sentiment_counts = df['sentimen'].value_counts()
                    st.bar_chart(sentiment_counts)
                    
                    # Hitung statistik untuk log
                    total_reviews = len(df)
                    # Gunakan .get() untuk menghindari error jika sentimen tertentu tidak ada
                    positive_count = sentiment_counts.get('positif', 0)
                    negative_count = sentiment_counts.get('negatif', 0)

                    # 5. Simpan Log ke Database
                    add_log(uploaded_file.name, int(total_reviews), int(positive_count), int(negative_count))
                    st.success("Analisis selesai dan log berhasil disimpan ke database!")

            else:
                st.error(f"Kolom '{column_name}' tidak ditemukan di dalam file CSV. Silakan periksa kembali.")

    except Exception as e:
        st.error(f"Terjadi kesalahan saat memproses file: {e}")

# Menampilkan Log Aktivitas dari Database
st.header("📜 Log Aktivitas Analisis")
conn = sqlite3.connect('analysis_log.db')
log_df = pd.read_sql_query("SELECT * FROM log ORDER BY analysis_time DESC", conn)
conn.close()
st.dataframe(log_df)