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

ASPEK = {
    "Tiket": ["tiket", "harga", "booking", "presale", "ots", "mahal", "murah", "dapat", "habis", "sold out", "sold", "telat", "kalah"],
    "Guest_star": ["guest star", "bintang tamu", "pengisi acara", "penampil", "band", "artis", "jkt48", "jkt", "oshi", "marsha", "ci shani", "shani", "zee", "amanda", "freya", "adel"],
    "Lokasi": ["venue", "lokasi", "tempat", "panggung", "stage", "semilir", "jawa", "jawa tengah", "semarang", "Semarang", "Bawen", "ungaran", "deket", "dekat", "jauh", "magelang", "wonosobo", "temanggung"]
}

@st.cache_resource

def extract_aspects(text, aspect_keywords):
    """Mengekstrak aspek yang ditemukan dalam teks."""
    found_aspects = []
    for aspect, keywords in aspect_keywords.items():
        for keyword in keywords:
            if keyword in text:
                found_aspects.append(aspect)
                break 
    return found_aspects

def load_model_and_vectorizer():
    """Memuat model SVM dan TF-IDF Vectorizer."""
    try:
        model = joblib.load('model_svm1.joblib')
        vectorizer = joblib.load('vectorizer1.joblib')
        return model, vectorizer
    except FileNotFoundError:
        st.error("File 'model_svm1.joblib' atau 'vectorizer1.joblib' tidak ditemukan. Pastikan file berada di folder yang sama.")
        st.stop()

def preprocess_text(text):
    """Membersihkan dan menstandarisasi teks input."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    text = re.sub(r'\@\w+|\#', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in list_stopwords]
    tokens = [stemmer.stem(word) for word in tokens]
    return " ".join(tokens)

try:
    stopwords.words('indonesian')
except LookupError:
    nltk.download('stopwords')

factory = StemmerFactory()
stemmer = factory.create_stemmer()
list_stopwords = set(stopwords.words('indonesian'))
model, vectorizer = load_model_and_vectorizer()

st.set_page_config(page_title="Analisis Sentimen SVM", layout="wide")
st.title("Aplikasi Analisis Sentimen Menggunakan SVM")
st.write("Unggah file CSV Anda, dan aplikasi ini akan memprediksi sentimen dari kolom teks yang Anda pilih menggunakan model SVM.")
st.markdown("---")

with st.sidebar:
    st.image("cropped-DUSEM-LOGO-512x512-12.png", use_container_width=True) 
    st.header("Upload Data Anda")
    uploaded_file = st.file_uploader("Pilih file CSV...", type=["csv"])

if uploaded_file is not None:
    try:
        df_input = pd.read_csv(uploaded_file)
        st.info(f"File **{uploaded_file.name}** berhasil diunggah ✅. Pratinjau data:")
        st.dataframe(df_input.head())

        available_columns = df_input.columns.tolist()
        text_column = st.selectbox("Pilih kolom yang berisi teks untuk dianalisis:", options=available_columns)

        analyze_button = st.button("Lakukan Analisis Sentimen", type="primary")

        if analyze_button:
            with st.spinner('Sedang menganalisis data... Proses ini mungkin memakan waktu beberapa saat.'):
                df_processed = df_input.copy()
                df_processed.dropna(subset=[text_column], inplace=True) 
                df_processed['text_cleaned'] = df_processed[text_column].apply(preprocess_text)
                features = vectorizer.transform(df_processed['text_cleaned'])
                predictions = model.predict(features)
                df_processed['prediksi_sentimen'] = predictions
                
                df_processed['aspek_ditemukan'] = df_processed['text_cleaned'].apply(lambda x: extract_aspects(x, ASPEK))
                
                aspek_sentimen_list = []
                for index, row in df_processed.iterrows():
                    sentimen = row['prediksi_sentimen']
                    aspek_list = row['aspek_ditemukan']
                    teks = row[text_column] 

                    if not aspek_list:
                        aspek_sentimen_list.append({
                            'teks': teks,
                            'sentimen': sentimen,
                            'aspek': 'general'                            
                        })
                    else:
                        for aspek in aspek_list:
                            aspek_sentimen_list.append({
                                'teks': teks,
                                'sentimen': sentimen,
                                'aspek': aspek
                            })
                
                df_aspek = pd.DataFrame(aspek_sentimen_list)
                
                aspek_summary = df_aspek.groupby(['aspek', 'sentimen']).size().reset_index(name='jumlah')
                
                st.markdown("---")
                st.header("📊 Hasil Analisis")
                st.subheader("Visualisasi Ringkasan Utama")
                total_data = len(df_processed)
                sentimen_counts = df_processed['prediksi_sentimen'].value_counts()
                sentimen_df = sentimen_counts.reset_index()
                sentimen_df.columns = ['sentimen', 'jumlah']
                positive_count = int(sentimen_counts.get('positif', 0))
                negative_count = int(sentimen_counts.get('negatif', 0))

                if total_data > 0:
                    positive_percentage = (positive_count / total_data) * 100
                    negative_percentage = (negative_count / total_data) * 100
                else:
                    positive_percentage = 0
                    negative_percentage = 0
                
                if positive_count > negative_count:
                    dominant_sentiment = "Cenderung Positif"
                elif negative_count > positive_count:
                    dominant_sentiment = "Cenderung Negatif"
                else:
                    dominant_sentiment = "Seimbang"

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Total Komentar Dianalisis",
                        value=f"{total_data} komentar"
                    )
                
                with col2:
                    st.metric(
                        label="Sentimen Umum",
                        value=dominant_sentiment
                    )
                    
                col_pos, col_neg = st.columns(2)
                with col_pos:
                    st.metric(
                        label="🟢 Sentimen Positif",
                        value=positive_count
                    )
                
                with col_neg:
                    st.metric(
                        label="🔴 Sentimen Negatif",
                        value=negative_count
                    )
                    
                st.markdown("---")
                col1, col2 = st.columns(2)
                positive_text = df_processed[df_processed['prediksi_sentimen'] == 'positif']['text_cleaned']
                negative_text = df_processed[df_processed['prediksi_sentimen'] == 'negatif']['text_cleaned']

                with col1:
                    st.subheader("Kata Kunci Utama")
                    st.markdown("##### 🟢 Kata Kunci Positif")
                    full_positive_text = " ".join(text for text in positive_text)

                    if full_positive_text.strip():
                        wordcloud_pos = WordCloud(width=600, height=200, background_color="white", colormap='Greens').generate(full_positive_text)
                        st.image(wordcloud_pos.to_array(), use_container_width=True)
                    else:
                        st.info("Tidak ada kata kunci positif yang ditemukan untuk divisualisasikan.")

                    st.markdown("##### 🔴 Kata Kunci Negatif")
                    full_negative_text = " ".join(text for text in negative_text)

                    if full_negative_text.strip():
                        wordcloud_neg = WordCloud(width=600, height=200, background_color="white", colormap='Reds').generate(full_negative_text)
                        st.image(wordcloud_neg.to_array(), use_container_width=True)
                    else:
                        st.info("Tidak ada kata kunci negatif yang ditemukan untuk divisualisasikan.")
                        
                with col2:
                    st.subheader("Distribusi Sentimen")
                    
                    fig = px.pie(
                        sentimen_df,
                        names='sentimen',
                        values='jumlah',
                        hole=0.5, 
                        color='sentimen',
                        color_discrete_map={
                            'positif': '#4CAF50',
                            'negatif': '#F44336',
                        }
                    )
                    
                    fig.update_traces(
                        textposition='inside', 
                        textinfo='percent+label',
                        marker=dict(line=dict(color='#FFFFFF', width=2))
                    )
                    
                    fig.update_layout(
                        showlegend=False,
                        margin=dict(t=0, b=0, l=0, r=0)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

                st.markdown("---")
                if not aspek_summary.empty:
                    st.subheader("Distribusi Sentimen per Aspek Spesifik")
                    
                    fig_aspek = px.bar(
                        aspek_summary,
                        x='aspek',
                        y='jumlah',
                        color='sentimen',
                        barmode='group',
                        title="Jumlah Sentimen Positif & Negatif untuk Setiap Aspek",
                        labels={'aspek': 'Aspek', 'jumlah': 'Jumlah Komentar', 'sentimen': 'Sentimen'},
                        color_discrete_map={
                            'positif': '#4CAF50',
                            'negatif': '#F44336'
                        }
                    )
                    fig_aspek.update_layout(xaxis_title="Aspek", yaxis_title="Jumlah Komentar")
                    st.plotly_chart(fig_aspek, use_container_width=True)
                else:
                    st.info("Tidak ditemukan aspek spesifik (seperti tiket atau guest star) dalam data komentar.")
                            
                st.markdown("---")
                st.header("Contoh Komentar Aktual")
                n_samples = 3 
                col1_comment, col2_comment = st.columns(2)

                with col1_comment:
                    st.subheader("🟢 Positif")
                    positive_comments = df_processed[df_processed['prediksi_sentimen'] == 'positif']
                    
                    if positive_comments.empty:
                        st.info("Tidak ada komentar positif yang ditemukan.")
                    else:
                        try:
                            samples = positive_comments.sample(n_samples)
                        except ValueError:
                            samples = positive_comments
                        
                        for _, row in samples.iterrows():
                            st.success(f"_{row[text_column]}_")

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
                            
                st.markdown("---")
                st.header("Wawasan & Kesimpulan")

                if not sentimen_counts.empty:
                    dominant_sentiment = sentimen_counts.idxmax()
                    dominant_count = sentimen_counts.max()
                    dominant_percentage = (dominant_count / total_data) * 100

                    if dominant_sentiment == 'positif':
                        tendency_text = "cenderung **positif**"
                    else: 
                        tendency_text = "cenderung **negatif**"

                    insight_text = f"""
                    Dari total **{total_data} komentar** yang dianalisis, respon audiens secara umum {tendency_text}. 

                    Sentimen **{dominant_sentiment.capitalize()}** menjadi yang paling menonjol, mencakup **{dominant_percentage:.1f}%** dari keseluruhan tanggapan.
                    """

                    st.info(insight_text)

                else:
                    st.warning("Tidak ada data yang dapat disimpulkan.")
                
                st.markdown("---")
                st.subheader("Tabel Data dengan Hasil Prediksi Sentimen")
                st.dataframe(df_aspek)
                
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")
else:
    st.warning("Silakan unggah file CSV melalui sidebar untuk memulai.")

st.markdown("---")
