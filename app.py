import streamlit as st
import pandas as pd
import joblib
import datetime

# --- VERİ VE MODELLERİ YÜKLEME ---
@st.cache_resource
def verileri_yukle():
    try:
        data = joblib.load('covid_project_data.pkl')
        return data
    except FileNotFoundError:
        st.error("Hata: 'covid_project_data.pkl' dosyası bulunamadı. Lütfen önce notebook'taki kayıt kodunu çalıştırın.")
        return None

data_artifacts = verileri_yukle()

# Sayfa Ayarları (Daha sade bir başlık)
st.set_page_config(page_title="Covid-19 Risk Tahmincisi", layout="centered")

if data_artifacts:
    models = data_artifacts['models']
    scaler = data_artifacts['scaler']
    imputer = data_artifacts['imputer']
    feature_names = data_artifacts['feature_names']

    # --- BAŞLIK ---
    st.title("🏥 Covid-19 Risk Analiz Sistemi")
    st.write("Aşağıdan kullanmak istediğiniz yapay zeka modelini seçin ve hasta bilgilerini girin.")
    st.divider()

    # --- 1. MODEL SEÇİMİ ---
    model_isimleri = list(models.keys())
    secilen_model_ismi = st.selectbox("📌 Tahmin İçin Kullanılacak Model:", model_isimleri)
    
    # Seçilen modeli değişkeni al
    aktif_model = models[secilen_model_ismi]

    st.info(f"Şu an **{secilen_model_ismi}** modeli ile analiz yapıyorsunuz.")
    st.divider()

    # --- 2. PARAMETRE GİRİŞLERİ (Form Yapısı) ---
    with st.form("tahmin_formu"):
        st.subheader("📝 Hasta Bilgileri")
        
        col1, col2 = st.columns(2)
        
        with col1:
            yas = st.number_input("Yaş", min_value=0, max_value=120, value=30)
            cinsiyet = st.selectbox("Cinsiyet", ["Erkek", "Kadın"])
            
            # Tarihsel veriler (Gecikme süresi hesaplamak için)
            st.write("🗓️ Tarih Bilgileri")
            sym_on = st.date_input("Semptom Başlangıç", datetime.date(2020, 1, 1))
            hosp_vis = st.date_input("Hastaneye Başvuru", datetime.date(2020, 1, 5))

        with col2:
            st.write("🤒 Klinik Bulgular")
            # Checkbox yerine Selectbox veya Radio daha şık durabilir, ama hızlı giriş için toggle iyidir.
            fever = st.toggle("Ateş (Fever)")
            cough = st.toggle("Öksürük (Cough)")
            tiredness = st.toggle("Yorgunluk (Tiredness)")
            
            # Buraya modelinizde olan diğer önemli semptomları ekleyebilirsiniz
            # Örneğin: difficulty_breathing = st.toggle("Nefes Darlığı")

        # Form Gönderme Butonu (En altta ortada)
        submit_btn = st.form_submit_button("ANALİZ ET VE SONUCU GÖSTER", use_container_width=True)

    # --- 3. TAHMİN İŞLEMİ ---
    if submit_btn:
        # Gecikme süresini hesapla
        delay_days = (hosp_vis - sym_on).days
        if delay_days < 0: delay_days = 0
        
        # Cinsiyet dönüşümü (Verisetinizdeki gibi)
        # Eğer 'Male'/'Female' ise İngilizceye çeviriyoruz
        cinsiyet_ing = "Male" if cinsiyet == "Erkek" else "Female"

        # Ham veriyi oluştur
        input_data = pd.DataFrame(0, index=[0], columns=feature_names)
        
        # Değerleri sütunlara yerleştir
        # NOT: Buradaki sütun isimleri 'feature_names' ile tam eşleşmeli.
        # Notebook'unuzdaki processed_data.columns listesine göre buraları kontrol edin.
        
        if 'age' in feature_names: input_data['age'] = yas
        if 'delay_days' in feature_names: input_data['delay_days'] = delay_days
        
        # One-Hot Encoding sütunları (Örn: gender_Male)
        col_gender = f"gender_{cinsiyet_ing}" 
        if col_gender in feature_names: input_data[col_gender] = 1
        
        # Semptomlar
        if 'fever' in feature_names: input_data['fever'] = 1 if fever else 0
        if 'cough' in feature_names: input_data['cough'] = 1 if cough else 0
        if 'tiredness' in feature_names: input_data['tiredness'] = 1 if tiredness else 0

        # İşle ve Tahmin Et
        try:
            input_imputed = imputer.transform(input_data)
            input_scaled = scaler.transform(input_imputed)
            
            tahmin = aktif_model.predict(input_scaled)[0]
            
            # Olasılık değeri varsa alalım
            if hasattr(aktif_model, "predict_proba"):
                olasilik = aktif_model.predict_proba(input_scaled)[0][1]
            else:
                olasilik = None

            # --- SONUÇ EKRANI ---
            st.markdown("---")
            if tahmin == 1:
                st.error("### ⚠️ SONUÇ: RİSKLİ (POZİTİF)")
                if olasilik:
                    st.write(f"Modelin ölüm riski tahmini: **%{olasilik*100:.1f}**")
                st.warning("Hastanın durumu kritik olabilir, ileri tetkik önerilir.")
            else:
                st.success("### ✅ SONUÇ: RİSK DÜŞÜK (NEGATİF)")
                if olasilik:
                    st.write(f"Modelin hayatta kalma tahmini: **%{(1-olasilik)*100:.1f}**")
                st.info("Hasta durumu stabil görünüyor.")

        except Exception as e:
            st.error(f"Tahmin sırasında bir hata oluştu: {e}")
            st.write("Detay: Sütun isimleri uyuşmuyor olabilir.")
