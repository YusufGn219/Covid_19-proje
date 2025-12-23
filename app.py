import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# --------------------------------------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------------------------------------
st.set_page_config(page_title="COVID-19 Risk Tahmini", page_icon="🦠", layout="wide")

st.markdown("<h1 style='color:#d63031; font-size:42px;'>🦠 COVID-19 Ölüm Riski Tahmini</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#636e72; font-size:16px;'>Hastanın semptom ve demografik bilgilerine göre risk analizi yapar.</div>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)

# --------------------------------------------------------------------------------
# MODEL YÜKLEME
# --------------------------------------------------------------------------------
@st.cache_resource
def load_project_data():
    model_path = "covid_project_data.pkl"
    if not os.path.exists(model_path):
        st.error(f"⚠️ '{model_path}' dosyası bulunamadı! Lütfen notebook dosyanızdaki son hücreyi çalıştırıp pkl dosyasını oluşturun.")
        return None
    return joblib.load(model_path)

project_data = load_project_data()

if project_data:
    models = project_data['models']
    scaler = project_data['scaler']
    imputer = project_data['imputer']
    feature_names = project_data['feature_names']
else:
    st.stop()

# --------------------------------------------------------------------------------
# SEÇENEK LİSTELERİ (Veri setinizden alınmıştır)
# --------------------------------------------------------------------------------
COUNTRIES = ['China', 'France', 'Japan', 'Malaysia', 'Nepal', 'Singapore', 'South Korea', 'Taiwan', 'Thailand', 'USA', 'Vietnam', 'Australia', 'Canada', 'Cambodia', 'Sri Lanka', 'Germany', 'Finland', 'UAE', 'Philippines', 'India', 'Italy', 'UK', 'Russia', 'Sweden', 'Spain', 'Belgium', 'Other']
LOCATIONS = ['Wuhan', 'Beijing', 'Shanghai', 'Guangdong', 'Other'] # Örnek olarak kısaltıldı, dilerseniz artırabilirsiniz.
GENDERS = ['male', 'female']
SYMPTOMS = ['fever', 'cough', 'sore throat', 'runny nose', 'dyspnea', 'pneumonia', 'headache', 'vomiting', 'diarrhea', 'fatigue', 'chill', 'body pain', 'malaise']

# --------------------------------------------------------------------------------
# YAN MENÜ: ANLIK TAHMİN
# --------------------------------------------------------------------------------
st.sidebar.header("📝 Hasta Bilgileri")

# Kullanıcıdan Girdiler
selected_model_name = st.sidebar.selectbox("Kullanılacak Model", list(models.keys()))
country = st.sidebar.selectbox("Ülke", COUNTRIES)
location = st.sidebar.selectbox("Bölge/Şehir", LOCATIONS)
gender = st.sidebar.selectbox("Cinsiyet", GENDERS)
age = st.sidebar.slider("Yaş", 0, 100, 35)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 📅 Tarih Bilgileri")
sym_on = st.sidebar.date_input("Semptom Başlangıç Tarihi", value=None)
hosp_vis = st.sidebar.date_input("Hastaneye Geliş Tarihi", value=None)

st.sidebar.markdown("---")
st.sidebar.markdown("##### ✈️ Seyahat ve Geçmiş")
vis_wuhan = st.sidebar.radio("Wuhan'ı Ziyaret Etti mi?", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")
from_wuhan = st.sidebar.radio("Wuhan'dan mı Geldi?", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")
recov = st.sidebar.radio("İyileşme Durumu (Recovered)", [0, 1], index=0, format_func=lambda x: "İyileşti" if x==1 else "Bilinmiyor/Hayır")

st.sidebar.markdown("---")
st.sidebar.markdown("##### 🤒 Semptomlar (Varsa Seçin)")
# Semptomları tek tek sormak yerine, veri setinizdeki column yapısına uygun olarak alıyoruz
symptom1 = st.sidebar.selectbox("Semptom 1", [""] + SYMPTOMS)
symptom2 = st.sidebar.selectbox("Semptom 2", [""] + SYMPTOMS)
symptom3 = st.sidebar.selectbox("Semptom 3", [""] + SYMPTOMS)
symptom4 = st.sidebar.selectbox("Semptom 4", [""] + SYMPTOMS)
symptom5 = st.sidebar.selectbox("Semptom 5", [""] + SYMPTOMS)
symptom6 = st.sidebar.selectbox("Semptom 6", [""] + SYMPTOMS)

# --------------------------------------------------------------------------------
# TAHMİN FONKSİYONU
# --------------------------------------------------------------------------------
def preprocess_input(input_dict):
    # DataFrame oluştur
    df = pd.DataFrame([input_dict])
    
    # Tarih Farkı (Delay Days) Hesabı
    if df['sym_on'][0] and df['hosp_vis'][0]:
        d1 = pd.to_datetime(df['sym_on'])
        d2 = pd.to_datetime(df['hosp_vis'])
        diff = (d2 - d1).dt.days
        df['delay_days'] = diff.clip(lower=0, upper=30)
    else:
        # Tarih girilmediyse ortalama bir değer veya 0 atayalım
        df['delay_days'] = 0 
    
    # Gereksiz sütunları düşür (Tarihler artık delay_days oldu)
    df = df.drop(columns=['sym_on', 'hosp_vis'], errors='ignore')
    
    # Categorical Encoding (Get Dummies)
    # Burada kritik nokta: Eğitimdeki sütun yapısını birebir oluşturmalıyız.
    df_encoded = pd.get_dummies(df)
    
    # Eğitim setindeki sütunlara göre hizala (Eksik sütunları 0 yap, fazlaları at)
    df_aligned = df_encoded.reindex(columns=feature_names, fill_value=0)
    
    # Imputer (Eksik Veri Doldurma)
    df_imputed = imputer.transform(df_aligned)
    
    # Scaler (Ölçeklendirme)
    df_scaled = scaler.transform(df_imputed)
    
    return df_scaled

# --------------------------------------------------------------------------------
# ANLIK TAHMİN BUTONU VE SONUÇ
# --------------------------------------------------------------------------------
st.subheader("🔍 Tekil Tahmin Sonucu")

if st.button("▶︎ Risk Durumunu Tahmin Et"):
    # Girdi sözlüğü
    input_data = {
        'location': location,
        'country': country,
        'gender': gender,
        'age': age,
        'sym_on': sym_on,
        'hosp_vis': hosp_vis,
        'vis_wuhan': vis_wuhan,
        'from_wuhan': from_wuhan,
        'recov': recov,
        'symptom1': symptom1 if symptom1 else np.nan,
        'symptom2': symptom2 if symptom2 else np.nan,
        'symptom3': symptom3 if symptom3 else np.nan,
        'symptom4': symptom4 if symptom4 else np.nan,
        'symptom5': symptom5 if symptom5 else np.nan,
        'symptom6': symptom6 if symptom6 else np.nan
    }
    
    try:
        X_pred = preprocess_input(input_data)
        model = models[selected_model_name]
        prediction = model.predict(X_pred)[0]
        
        # Sonuç Görselleştirme
        if prediction == 1:
            st.error(f"⚠️ Tahmin: **Yüksek Risk / Ölüm (1)**")
            st.markdown("Model bu hastanın durumunu kritik olarak değerlendirdi.")
        else:
            st.success(f"✅ Tahmin: **Düşük Risk / İyileşme (0)**")
            st.markdown("Model bu hastanın iyileşmesini öngörüyor.")
            
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")

# --------------------------------------------------------------------------------
# TOPLU TAHMİN (CSV YÜKLEME)
# --------------------------------------------------------------------------------
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)
st.subheader("📂 Toplu Tahmin (CSV Yükleme)")
st.markdown("<div style='color:#636e72; font-size:14px;'>Eğitim veri setinizdeki formatta (age, gender, country vb.) bir CSV yükleyin.</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Dosya Seçin", type=["csv"])

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Yüklenen Veri (İlk 5 satır):")
        st.dataframe(data.head())
        
        if st.button("Tüm Listeyi Tahmin Et"):
            # Veri Ön İşleme (Notebook mantığının aynısı)
            data_proc = data.copy()
            
            # Tarih dönüşümleri
            if 'sym_on' in data_proc.columns and 'hosp_vis' in data_proc.columns:
                data_proc['sym_on'] = pd.to_datetime(data_proc['sym_on'], errors='coerce')
                data_proc['hosp_vis'] = pd.to_datetime(data_proc['hosp_vis'], errors='coerce')
                data_proc['delay_days'] = (data_proc['hosp_vis'] - data_proc['sym_on']).dt.days
                data_proc['delay_days'] = data_proc['delay_days'].clip(lower=0, upper=30)
                # NaN delay_days doldur
                data_proc['delay_days'].fillna(0, inplace=True)
                data_proc.drop(columns=['sym_on', 'hosp_vis'], inplace=True)
            
            if 'id' in data_proc.columns:
                data_proc.drop(columns=['id'], inplace=True)
                
            # Encoding & Scaling
            data_encoded = pd.get_dummies(data_proc)
            data_aligned = data_encoded.reindex(columns=feature_names, fill_value=0)
            data_imputed = imputer.transform(data_aligned)
            data_scaled = scaler.transform(data_imputed)
            
            # Tahmin
            model = models[selected_model_name]
            predictions = model.predict(data_scaled)
            
            data['Tahmin_Sonucu'] = predictions
            data['Tahmin_Sonucu'] = data['Tahmin_Sonucu'].map({0: 'İyileşme', 1: 'Ölüm/Risk'})
            
            st.success("✅ Tahminler tamamlandı!")
            st.dataframe(data)
            
            # İndirme Butonu
            csv = data.to_csv(index=False).encode('utf-8')
            st.download_button("📤 Sonuçları İndir (CSV)", csv, "tahmin_sonuclari.csv")
            
    except Exception as e:
        st.error(f"Hata: {str(e)}")
