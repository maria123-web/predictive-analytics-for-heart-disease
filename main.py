import streamlit as st
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("model_rf.joblib")
scaler = joblib.load("scaler.joblib")

# Judul aplikasi
st.title("💓 Prediksi Penyakit Jantung")
st.write("Masukkan data pasien di bawah ini untuk memprediksi risiko penyakit jantung berdasarkan parameter medis.")

# 🧍‍♂️ Input nama pasien
nama = st.text_input("Nama Pasien", placeholder="Masukkan nama lengkap Anda")

# Input data medis
age = st.number_input("Umur (tahun)", min_value=1, max_value=120, value=30)
sex = st.selectbox("Jenis Kelamin", ("Laki-laki", "Perempuan"))

cp_dict = {
    "Typical angina (nyeri dada khas)": 0,
    "Atypical angina (nyeri dada tidak khas)": 1,
    "Non-anginal pain (bukan nyeri dada angina)": 2,
    "Asymptomatic (tanpa gejala nyeri dada)": 3
}
cp_choice = st.selectbox("Tipe Nyeri Dada (cp)", list(cp_dict.keys()))
cp = cp_dict[cp_choice]

trestbps = st.number_input("Tekanan Darah Istirahat (trestbps) — mm Hg", min_value=50, max_value=250, value=120)
chol = st.number_input("Kolesterol (chol) — mg/dl", min_value=100, max_value=600, value=200)
fbs = st.selectbox("Gula Darah Puasa > 120 mg/dl (fbs)", ["Tidak (0)", "Ya (1)"])
fbs = 1 if "Ya" in fbs else 0

restecg_dict = {
    "Normal": 0,
    "ST-T wave abnormality": 1,
    "Menunjukkan hipertrofi ventrikel kiri": 2
}
restecg_choice = st.selectbox("Hasil EKG Istirahat (restecg)", list(restecg_dict.keys()))
restecg = restecg_dict[restecg_choice]

thalach = st.number_input("Denyut Jantung Maksimum (thalach)", min_value=50, max_value=250, value=150)
exang = st.selectbox("Nyeri Dada Akibat Olahraga (exang)", ["Tidak (0)", "Ya (1)"])
exang = 1 if "Ya" in exang else 0

oldpeak = st.number_input("Penurunan ST (oldpeak) relatif terhadap istirahat", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope_dict = {
    "Upsloping (menanjak)": 0,
    "Flat (datar)": 1,
    "Downsloping (menurun)": 2
}
slope_choice = st.selectbox("Kemiringan Segmen ST (slope)", list(slope_dict.keys()))
slope = slope_dict[slope_choice]

ca_dict = {
    "0 - Tidak ada pembuluh darah tersumbat": 0,
    "1 - 1 pembuluh darah utama tersumbat": 1,
    "2 - 2 pembuluh darah utama tersumbat": 2,
    "3 - 3 pembuluh darah utama tersumbat": 3,
    "4 - 4 pembuluh darah utama tersumbat": 4
}
ca_choice = st.selectbox("Jumlah Pembuluh Darah Utama yang Terlihat (ca)", list(ca_dict.keys()))
ca = ca_dict[ca_choice]

thal_dict = {
    "Normal": 0,
    "Fixed defect (kerusakan tetap)": 1,
    "Reversible defect (kerusakan reversibel)": 2,
    "Unknown / missing": 3
}
thal_choice = st.selectbox("Thalassemia (thal)", list(thal_dict.keys()))
thal = thal_dict[thal_choice]

# Konversi gender ke numerik
sex_value = 1 if sex == "Laki-laki" else 0

# Data inputan user
features = np.array([[age, sex_value, cp, trestbps, chol, fbs, restecg,
                      thalach, exang, oldpeak, slope, ca, thal]])

# Normalisasi dengan scaler
features_scaled = scaler.transform(features)

# Tombol prediksi
if st.button("Prediksi"):
    if nama.strip() == "":
        st.warning("⚠️ Mohon masukkan nama pasien terlebih dahulu sebelum melakukan prediksi.")
    else:
        st.subheader("Hasil Prediksi:")
        prediction = model.predict(features_scaled)
        if prediction[0] == 1:
            result = f"💔  **{nama}** berisiko terkena penyakit jantung."
            st.error(result)
        else:
            result = f"💖  **{nama}** tidak berisiko terkena penyakit jantung."
            st.success(result)
        
        

# Footer
st.markdown("---")
st.caption("Model: KNN | Dataset: Heart Disease (Kaggle) | Dibuat dengan Streamlit 💕")
