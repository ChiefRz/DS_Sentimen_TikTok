import streamlit as st
import pandas as pd
import joblib
import re
import nltk
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud
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
                
                
                # Tahap 11: Visualisasi
                st.subheader("Visualisasi Ringkasan Utama")
                total_data = len(df_processed)
                sentimen_counts = df_processed['prediksi_sentimen'].value_counts()
                sentimen_df = sentimen_counts.reset_index()
                sentimen_df.columns = ['sentimen', 'jumlah'] 
                
                col1, col2 = st.columns([2, 3]) # Membuat 2 kolom dengan rasio lebar 1:2
                with col1:
                    st.markdown("#### Metrik Utama")
                    st.metric(label="Total Data Dianalisis", value=f"{total_data} Komentar")
                    # Menampilkan persentase untuk setiap sentimen
                    for sentimen in sentimen_df['sentimen']:
                        count = sentimen_df[sentimen_df['sentimen'] == sentimen]['jumlah'].iloc[0]
                        percentage = (count / total_data) * 100
                        st.write(f"**{sentimen.capitalize()}:** {count} komentar ({percentage:.1f}%)")
                        
                with col2:
                    st.markdown("#### Distribusi Sentimen")
                    
                    # Membuat Donut Chart dengan Plotly
                    fig = px.pie(
                        sentimen_df,
                        names='sentimen',
                        values='jumlah',
                        hole=0.5, # Ini yang membuat pie chart menjadi donut char
                        color='sentimen',
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
                
                # =================================================================
                # Tahap 11.2: Visualisasi Kata Kunci (Word Cloud)
                # =================================================================
                st.markdown("---")
                st.header("Kata Kunci Dominan Berdasarkan Sentimen")

                # 1. Filter data berdasarkan sentimen dari kolom 'text_cleaned'
                positive_text = df_processed[df_processed['prediksi_sentimen'] == 'positif']['text_cleaned']
                negative_text = df_processed[df_processed['prediksi_sentimen'] == 'negatif']['text_cleaned']

                # 2. Buat dua kolom untuk tata letak
                col1_wc, col2_wc = st.columns(2)

                # 3. Buat word cloud untuk sentimen positif
                with col1_wc:
                    st.subheader("🟢 Kata Kunci Positif")
                    full_positive_text = " ".join(text for text in positive_text)

                    if full_positive_text.strip():
                        wordcloud_pos = WordCloud(width=800, height=400, background_color="white", colormap='Greens').generate(full_positive_text)
                        st.image(wordcloud_pos.to_array(), use_container_width=True)
                    else:
                        st.info("Tidak ada kata kunci positif yang ditemukan untuk divisualisasikan.")

                    # 4. Buat word cloud untuk sentimen negatif
                with col2_wc:
                    st.subheader("🔴 Kata Kunci Negatif")
                    full_negative_text = " ".join(text for text in negative_text)

                    if full_negative_text.strip():
                        wordcloud_neg = WordCloud(width=800, height=400, background_color="white", colormap='Reds').generate(full_negative_text)
                        st.image(wordcloud_neg.to_array(), use_container_width=True)
                    else:
                        st.info("Tidak ada kata kunci negatif yang ditemukan untuk divisualisasikan.")
                    # =================================================================
                    # AKHIR DARI BLOK WORD CLOUD
                    # =================================================================            

                # =================================================================
                # Tahap 11.3: Menampilkan Contoh Komentar Aktual
                # =================================================================
                st.markdown("---")
                st.header("💬 Contoh Komentar Aktual")

                # Jumlah komentar yang ingin ditampilkan per kategori
                n_samples = 3 

                # Membuat tiga kolom
                col1_comment, col2_comment = st.columns(2)

                # --- Kolom Komentar Positif ---
                with col1_comment:
                    st.subheader("🟢 Positif")
                    positive_comments = df_processed[df_processed['prediksi_sentimen'] == 'positif']
                    
                    if positive_comments.empty:
                        st.info("Tidak ada komentar positif yang ditemukan.")
                    else:
                        # Ambil sampel acak, jika jumlah data kurang, ambil semuanya
                        try:
                            samples = positive_comments.sample(n_samples)
                        except ValueError:
                            samples = positive_comments
                        
                        # Tampilkan setiap sampel menggunakan st.success
                        for _, row in samples.iterrows():
                            st.success(f"_{row[text_column]}_")

                # --- Kolom Komentar Negatif ---
                with col2_comment:
                    st.subheader("🔴 Negatif")
                    negative_comments = df_processed[df_processed['prediksi_sentimen'] == 'negatif']

                    if negative_comments.empty:
                        st.info("Tidak ada komentar negatif yang ditemukan.")
                    else:
                        try:
                            samples = negative_comments.sample(n_samples)
                        except ValueError:
                            samples = negative_comments

                        for _, row in samples.iterrows():
                            st.error(f"_{row[text_column]}_")
                # =================================================================
                # AKHIR DARI BLOK CONTOH KOMENTAR
                # =================================================================
                # =================================================================
                # Tahap 11.4: Menampilkan Wawasan dan Kesimpulan
                # =================================================================
                st.markdown("---")
                st.header("💡 Wawasan & Kesimpulan")

                # Pastikan sentimen_counts tidak kosong
                if not sentimen_counts.empty:
                    # 1. Cari sentimen yang paling dominan (total_data sudah dihitung sebelumnya)
                    dominant_sentiment = sentimen_counts.idxmax()
                    dominant_count = sentimen_counts.max()
                    dominant_percentage = (dominant_count / total_data) * 100

                    # 2. Tentukan kalimat berdasarkan sentimen dominan
                    if dominant_sentiment == 'positif':
                        tendency_text = "cenderung **positif**"
                        icon = "✅"
                    else: 
                        tendency_text = "cenderung **negatif**"
                        icon = "❌"

                    # 3. Rakit kalimat kesimpulan secara dinamis
                    insight_text = f"""
                    {icon} Dari total **{total_data} komentar** yang dianalisis, respon audiens secara umum {tendency_text}. 

                    Sentimen **{dominant_sentiment.capitalize()}** menjadi yang paling menonjol, mencakup **{dominant_percentage:.1f}%** dari keseluruhan tanggapan.
                    """

                    # 4. Tampilkan kesimpulan dalam kotak info
                    st.info(insight_text)

                else:
                    st.warning("Tidak ada data yang dapat disimpulkan.")
                # =================================================================
                # AKHIR DARI BLOK KESIMPULAN
                # =================================================================
                st.markdown("---")
                st.header("📊 Hasil Analisis")
                st.subheader("Tabel Data dengan Hasil Prediksi Sentimen")
                st.dataframe(df_processed[[text_column, 'prediksi_sentimen']])
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.warning("Silakan unggah file CSV melalui sidebar untuk memulai.")

st.markdown("---")
st.write("Dibuat menggunakan Streamlit.")