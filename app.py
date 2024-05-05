import streamlit as st
import joblib
import time

# define model requirement
model =  joblib.load("klasifikasi.joblib")
vectorizer = joblib.load("vectorizerKlasifikasi.joblib")

# Menyetel tema Streamlit
st.set_page_config(page_title="Klasifikasi Bahasa Bugis Makassar", layout="wide", initial_sidebar_state="expanded")

# Judul dan deskripsi
st.title("Klasifikasi Bahasa Bugis Makassar")

# Input teks dan tombol prediksi
with st.container():
    sentences = st.text_area("Masukkan kalimat untuk diklasifikasi", height=100)
    button = st.button("Klasifikasi")

# Tampilan hasil prediksi
with st.container(border=True):
    if button:
        with st.spinner('Menganalisis...'):
            # Mengubah kalimat menjadi vektor menggunakan vectorizer
            vectorized = vectorizer.transform([sentences])

            # Memprediksi
            predicted = model.predict(vectorized)[0]

            # Menghitung probabilitas
            probabilities = model.predict_proba(vectorized)[0]
            probabilities = [f"{round(x*100, 2)}%" for x in probabilities]

            probability = {
                "Bugis" : probabilities[0],
                "Makassar" : probabilities[1],
            }

            time.sleep(2)

            # Menampilkan prediksi dan probabilitas
            st.subheader("Prediksi:")
            if predicted == "Positif":
                st.success(predicted)
            else:
                st.error(predicted)

            st.subheader("Probabilitas:")
            st.table(probability)
