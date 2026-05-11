import streamlit as st
import pandas as pd
import numpy as np
import networkx as nx
import nltk
import re
from nltk.tokenize import sent_tokenize
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. Konfigurasi Halaman
st.set_page_config(page_title="Text Summarization NLP", layout="wide")


st.title("Aplikasi Peringkasan Berita Otomatis (TextRank)")
st.write("Dibangun menggunakan pendekatan *Extractive Summarization* (TF-IDF & Cosine Similarity) ")
st.markdown("---")

# 2. Caching Model ( st.spinner agar ada indikator loading)
@st.cache_resource
def load_nlp_components():
    nltk.download('punkt')
    nltk.download('punkt_tab')
    stemmer = StemmerFactory().create_stemmer()
    stopword = StopWordRemoverFactory().create_stop_word_remover()
    return stemmer, stopword

# Panggil fungsinya 
with st.spinner("Sedang memuat model NLP & Kamus Bahasa Indonesia (Sastrawi)... Mohon tunggu sebentar."):
    stemmer, stopword = load_nlp_components()

# 3. Fungsi Pembersihan
def bersihkan_teks_awal(teks):
    teks_bersih = re.sub(r'\([a-zA-Z]+/[a-zA-Z]+\)', '', teks)
    teks_bersih = re.sub(r'\s+', ' ', teks_bersih)
    return teks_bersih.strip()

def bersihkan_kalimat(kalimat):
    teks = kalimat.lower()
    teks = re.sub(r'[^a-z\s]', '', teks) 
    teks = stopword.remove(teks)         
    teks = stemmer.stem(teks)            
    return teks

# 4. Fungsi Utama TextRank
def jalankan_textrank(teks_asli, jumlah_kalimat_ringkasan):
    teks_siap_proses = bersihkan_teks_awal(teks_asli)
    kalimat_asli = sent_tokenize(teks_siap_proses)
    
    if len(kalimat_asli) <= jumlah_kalimat_ringkasan:
        return teks_siap_proses, None
        
    kalimat_bayangan = [bersihkan_kalimat(k) for k in kalimat_asli]
    
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(kalimat_bayangan)
    similarity_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    nx_graph = nx.from_numpy_array(similarity_matrix)
    skor_kalimat = nx.pagerank(nx_graph)
    
    ranked_sentences = sorted(((skor_kalimat[i], i) for i in range(len(kalimat_asli))), reverse=True)
    
    data_skor = []
    for i in range(len(ranked_sentences)):
        skor, idx = ranked_sentences[i]
        data_skor.append({
            "Peringkat": i + 1,
            "Skor": round(skor, 4),
            "Indeks": idx,
            "Kalimat Asli": kalimat_asli[idx]
        })
    df_skor = pd.DataFrame(data_skor)
    
    top_indices = [ranked_sentences[i][1] for i in range(jumlah_kalimat_ringkasan)]
    top_indices.sort() 
    
    summary_sentences = [kalimat_asli[i] for i in top_indices]
    return " ".join(summary_sentences), df_skor

# 5. Area Input (Lanjutan UI)
input_teks = st.text_area("Masukkan teks berita berbahasa Indonesia di sini:", height=250)

col1, col2 = st.columns([1, 3])
with col1:
    jumlah_kalimat = st.number_input("Jumlah kalimat ringkasan:", min_value=1, max_value=10, value=3)
with col2:
    st.write("") 
    st.write("")
    tombol_ringkas = st.button("Ringkas Teks Sekarang", type="primary")

st.markdown("---")

# Logika Eksekusi
if tombol_ringkas:
    if not input_teks.strip():
        st.warning("Silakan masukkan teks berita terlebih dahulu.")
    else:
        with st.spinner('Mesin sedang menganalisis kalimat...'):
            hasil, df_skor = jalankan_textrank(input_teks, jumlah_kalimat)
            
            st.subheader("Hasil Ringkasan:")
            st.info(hasil)
            
            if df_skor is not None:
                with st.expander("Lihat Analisis Skor PageRank per Kalimat"):
                    st.dataframe(df_skor, use_container_width=True)
