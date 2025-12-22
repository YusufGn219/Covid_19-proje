import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import datetime

# 1. Kaydettiğimiz proje verilerini geri yüklüyoruz
@st.cache_resource
def verileri_yukle():
    try:
        data = joblib.load('covid_project_data.pkl')
        return data
    except FileNotFoundError:
        st.error("Lütfen 'covid_project_data.pkl' dosyasının aynı klasörde olduğundan emin olun.")
        return None

data_artifacts = verileri_yukle()

# Sayfa Ayarları
st.set_page_config(page_title="Covid-19 Tahmin Paneli", layout="wide", page_icon="🦠")

if data_artifacts:
    models = data_artifacts['models']
    scaler = data_artifacts['scaler']
    imputer = data_artifacts['imputer']
    feature_names = data_artifacts['feature_names']
    X_test_saved = data_artifacts['X_test']
    y_test_saved = data_artifacts['y_test']

    st.title("🦠 Covid-19 Klinik Tahmin ve Analiz Paneli")
    st.markdown("""
    Bu sistem, hastanın semptomlarına ve demografik verilerine dayanarak risk tahmini yapar.
    Ayrıca eğitilen modellerin başarı performanslarını (Confusion Matrix) karşılaştırır.
    """)

    # --- SİDEBAR: VERİ GİRİŞİ ---
    st.sidebar.header("📝 Hasta Bilgileri")

    # Not: Sütun isimlerini kendi verisetinizdeki orijinal isimlere göre kontrol edin!
    # Bu örnekte genel Covid veri setleri baz alınmıştır.
    
    col_s1, col_s2 = st.sidebar.columns(2)
    with col_s1:
        yas = st.number_input("Yaş", 0, 120, 45)
    with col_s2:
        cinsiyet = st.selectbox("Cinsiyet", ["Male", "Female"])

    # Tarihsel verilerden 'delay_days' hesaplama (Notebook'taki mantık)
    st.sidebar.subheader("Tarih Bilgileri")
    sym_on = st.sidebar.date_input("Semptom Başlangıç Tarihi", datetime.date(2020, 1, 1))
    hosp_vis = st.sidebar.date_input("Hastaneye Başvuru Tarihi", datetime.date(2020, 1, 5))

    # Semptomlar
    st.sidebar.subheader("Klinik Bulgular")
    fever = st.sidebar.checkbox("Ateş (Fever)")
    cough = st.sidebar.checkbox("Öksürük (Cough)")
    tiredness = st.sidebar.checkbox("Yorgunluk (Tiredness)")
    # İhtiyaca göre diğer semptomları ekleyebilirsiniz...

    tahmin_btn = st.sidebar.button("Sonucu Tahmin Et", type="primary")

    # --- ORTA KISIM: MODEL PERFORMANSLARI ---
    st.header("📊 Model Başarı Analizi (Confusion Matrix)")
    st.info("Aşağıdaki grafikler, modellerin test verisi üzerindeki gerçek performansını gösterir.")

    col1, col2, col3 = st.columns(3)
    cols_list = [col1, col2, col3]

    # Modelleri döngüyle çizdir
    for i, (name, model) in enumerate(models.items()):
        with cols_list[i % 3]:
            # Test verisiyle tahmin yap
            y_pred_test = model.predict(X_test_saved)
            cm = confusion_matrix(y_test_saved, y_pred_test)

            # Grafiği çiz
            fig, ax = plt.subplots(figsize=(4, 3))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax)
            ax.set_title(f"{name}", fontsize=10)
            ax.set_ylabel("Gerçek")
            ax.set_xlabel("Tahmin")
            st.pyplot(fig)

    # --- TAHMİN BÖLÜMÜ ---
    if tahmin_btn:
        st.divider()
        st.subheader("🔍 Tahmin Sonucu")

        # 1. 'delay_days' Hesaplama
        delay_days = (hosp_vis - sym_on).days
        if delay_days < 0:
            delay_days = 0  # Hatalı tarih girişini engelle

        # 2. Ham Veri Sözlüğü Oluşturma
        # Buradaki anahtarlar (keys), One-Hot Encoding öncesi kolonlara benzemeli veya
        # doğrudan modelin beklediği özelliklere dönüştürülmeli.
        
        # En güvenli yöntem: Tüm özelliklerin olduğu boş bir DataFrame yaratıp içini doldurmak.
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)

        # Değerleri Doldurma (Burası Feature Engineering kısmıdır)
        # Eğer 'age' sütunu varsa:
        if 'age' in feature_names:
            input_data['age'] = yas
        
        # Eğer 'delay_days' varsa:
        if 'delay_days' in feature_names:
            input_data['delay_days'] = delay_days

        # Kategorik veriler (One-Hot Encoded sütunlar için)
        # Örnek: Eğer sütun adı 'gender_Male' ise:
        if f'gender_{cinsiyet}' in feature_names:
            input_data[f'gender_{cinsiyet}'] = 1
        
        # Semptomlar (Eğer sütunlar 'fever', 'cough' gibi direkt isimlerse)
        if 'fever' in feature_names: input_data['fever'] = 1 if fever else 0
        if 'cough' in feature_names: input_data['cough'] = 1 if cough else 0
        if 'tiredness' in feature_names: input_data['tiredness'] = 1 if tiredness else 0

        # Not: Notebook'unuzda sütun isimleri farklıysa (örn: 'symptom1', 'symptom2')
        # yukarıdaki atamaları o isimlere göre düzeltmelisiniz.

        try:
            # 3. Eksik Veri Tamamlama (Imputer)
            input_imputed = imputer.transform(input_data)

            # 4. Ölçeklendirme (Scaler)
            input_scaled = scaler.transform(input_imputed)

            # 5. Tahmin (En iyi model ile, örneğin XGBoost)
            secilen_model = models.get('XGBoost', list(models.values())[0])
            tahmin = secilen_model.predict(input_scaled)[0]
            olasilik = secilen_model.predict_proba(input_scaled)[0][1] if hasattr(secilen_model, "predict_proba") else 0

            # Sonucu Göster
            col_res1, col_res2 = st.columns([1, 2])
            with col_res1:
                if tahmin == 1:
                    st.error("⚠️ YÜKSEK RİSK")
                    st.write(f"Ölüm Riski Olasılığı: **%{olasilik*100:.2f}**")
                else:
                    st.success("✅ DÜŞÜK RİSK")
                    st.write(f"Hayatta Kalma Olasılığı: **%{(1-olasilik)*100:.2f}**")
            
            with col_res2:
                st.info(f"Model ({type(secilen_model).__name__}) bu hastanın semptomlarına göre yukarıdaki tahmini yapmıştır.")
                st.write(f"Hesaplanan Gecikme Süresi: {delay_days} gün")

        except Exception as e:
            st.error(f"Tahmin sırasında hata oluştu: {e}")
            st.warning("Lütfen 'feature_names' ile 'input_data' sütunlarının eşleştiğinden emin olun.")
