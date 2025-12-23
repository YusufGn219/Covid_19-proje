import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sklearn
from datetime import date

# --------------------------------------------------------------------------------
# SAYFA AYARLARI
# --------------------------------------------------------------------------------
st.set_page_config(page_title="COVID-19 Risk Tahmini", page_icon="🦠", layout="wide")

st.markdown("<h1 style='color:#d63031; font-size:42px;'>🦠 COVID-19 Ölüm Riski Tahmini</h1>", unsafe_allow_html=True)
st.markdown("<div style='color:#636e72; font-size:16px;'>Hastanın semptom ve demografik bilgilerine göre risk analizi yapar.</div>", unsafe_allow_html=True)
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)

# Debug: sürümler
with st.expander("🔧 Ortam Bilgisi (Debug)", expanded=False):
    st.write("scikit-learn:", sklearn.__version__)

# --------------------------------------------------------------------------------
# MODEL (PIPELINE) YÜKLEME
# --------------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    model_path = "covid_pipeline.pkl"
    if not os.path.exists(model_path):
        st.error(
            f"⚠️ '{model_path}' dosyası bulunamadı!\n\n"
            "Notebook tarafında Pipeline'ı kaydedip bu dosyayı proje köküne koymalısın."
        )
        return None
    try:
        pipe = joblib.load(model_path)
        return pipe
    except Exception as e:
        st.error(f"⚠️ Model yüklenirken hata oluştu: {e}")
        return None

pipe = load_pipeline()
if pipe is None:
    st.stop()

# --------------------------------------------------------------------------------
# SEÇENEK LİSTELERİ
# --------------------------------------------------------------------------------
COUNTRIES = [
    'China', 'France', 'Japan', 'Malaysia', 'Nepal', 'Singapore', 'South Korea',
    'Taiwan', 'Thailand', 'USA', 'Vietnam', 'Australia', 'Canada', 'Cambodia',
    'Sri Lanka', 'Germany', 'Finland', 'UAE', 'Philippines', 'India', 'Italy',
    'UK', 'Russia', 'Sweden', 'Spain', 'Belgium', 'Other'
]
LOCATIONS = ['Wuhan', 'Beijing', 'Shanghai', 'Guangdong', 'Other']
GENDERS = ['male', 'female']
SYMPTOMS = [
    'fever', 'cough', 'sore throat', 'runny nose', 'dyspnea', 'pneumonia',
    'headache', 'vomiting', 'diarrhea', 'fatigue', 'chill', 'body pain', 'malaise'
]

# --------------------------------------------------------------------------------
# SIDEBAR: GİRDİLER
# --------------------------------------------------------------------------------
st.sidebar.header("📝 Hasta Bilgileri")

country = st.sidebar.selectbox("Ülke", COUNTRIES, index=COUNTRIES.index("Other") if "Other" in COUNTRIES else 0)
location = st.sidebar.selectbox("Bölge/Şehir", LOCATIONS, index=LOCATIONS.index("Other") if "Other" in LOCATIONS else 0)
gender = st.sidebar.selectbox("Cinsiyet", GENDERS, index=0)
age = st.sidebar.slider("Yaş", 0, 100, 35)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 📅 Tarih Bilgileri")
use_dates = st.sidebar.checkbox("Tarih bilgisi gireceğim", value=False)

sym_on = None
hosp_vis = None
if use_dates:
    sym_on = st.sidebar.date_input("Semptom Başlangıç Tarihi", value=date(2020, 1, 1))
    hosp_vis = st.sidebar.date_input("Hastaneye Geliş Tarihi", value=date(2020, 1, 2))

st.sidebar.markdown("---")
st.sidebar.markdown("##### ✈️ Seyahat ve Geçmiş")
vis_wuhan = st.sidebar.radio("Wuhan'ı Ziyaret Etti mi?", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")
from_wuhan = st.sidebar.radio("Wuhan'dan mı Geldi?", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")
recov = st.sidebar.radio("İyileşme Durumu (Recovered)", [0, 1], index=0, format_func=lambda x: "İyileşti" if x == 1 else "Bilinmiyor/Hayır")

st.sidebar.markdown("---")
st.sidebar.markdown("##### 🤒 Semptomlar (Varsa Seçin)")
symptom1 = st.sidebar.selectbox("Semptom 1", [""] + SYMPTOMS)
symptom2 = st.sidebar.selectbox("Semptom 2", [""] + SYMPTOMS)
symptom3 = st.sidebar.selectbox("Semptom 3", [""] + SYMPTOMS)
symptom4 = st.sidebar.selectbox("Semptom 4", [""] + SYMPTOMS)
symptom5 = st.sidebar.selectbox("Semptom 5", [""] + SYMPTOMS)
symptom6 = st.sidebar.selectbox("Semptom 6", [""] + SYMPTOMS)

# --------------------------------------------------------------------------------
# ÖN İŞLEME (HAM GİRDİYİ PIPELINE'A HAZIRLAMA)
# --------------------------------------------------------------------------------
def build_input_row():
    # Date -> delay_days (Notebook’ta yaptığın mantık)
    delay_days = 0
    if use_dates and sym_on is not None and hosp_vis is not None:
        d1 = pd.to_datetime(sym_on)
        d2 = pd.to_datetime(hosp_vis)
        diff = (d2 - d1).days
        delay_days = int(np.clip(diff, 0, 30)) if diff is not None else 0

    row = {
        "location": location,
        "country": country,
        "gender": gender,
        "age": age,
        "delay_days": delay_days,
        "vis_wuhan": vis_wuhan,
        "from_wuhan": from_wuhan,
        "recov": recov,
        "symptom1": (symptom1 if symptom1 else np.nan),
        "symptom2": (symptom2 if symptom2 else np.nan),
        "symptom3": (symptom3 if symptom3 else np.nan),
        "symptom4": (symptom4 if symptom4 else np.nan),
        "symptom5": (symptom5 if symptom5 else np.nan),
        "symptom6": (symptom6 if symptom6 else np.nan),
    }
    return pd.DataFrame([row])

# --------------------------------------------------------------------------------
# TEKİL TAHMİN
# --------------------------------------------------------------------------------
st.subheader("🔍 Tekil Tahmin Sonucu")

colA, colB = st.columns([1, 1], vertical_alignment="top")
with colA:
    st.markdown("**Girdi Özeti**")
    preview_df = build_input_row()
    st.dataframe(preview_df, use_container_width=True)

with colB:
    st.markdown("**Tahmin**")
    if st.button("▶︎ Risk Durumunu Tahmin Et"):
        try:
            X_pred = build_input_row()
            pred = pipe.predict(X_pred)[0]

            # Eğer olasılık varsa göster
            proba_text = ""
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(X_pred)[0]
                # Varsayım: sınıf 1 = ölüm/risk
                if len(proba) >= 2:
                    proba_text = f" (Risk olasılığı: **%{proba[1]*100:.1f}**)"
            
            if int(pred) == 1:
                st.error(f"⚠️ Tahmin: **Yüksek Risk / Ölüm (1)**{proba_text}")
                st.markdown("Model bu hastanın durumunu kritik olarak değerlendirdi.")
            else:
                st.success(f"✅ Tahmin: **Düşük Risk / İyileşme (0)**{proba_text}")
                st.markdown("Model bu hastanın iyileşmesini öngörüyor.")
        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

# --------------------------------------------------------------------------------
# TOPLU TAHMİN (CSV)
# --------------------------------------------------------------------------------
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)
st.subheader("📂 Toplu Tahmin (CSV Yükleme)")
st.markdown(
    "<div style='color:#636e72; font-size:14px;'>"
    "CSV içinde en az şu kolonlar olmalı: age, gender, country, location, vis_wuhan, from_wuhan, recov "
    "ve (varsa) sym_on + hosp_vis veya direkt delay_days."
    "</div>",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Dosya Seçin", type=["csv"])

def ensure_delay_days(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()

    # Eğer delay_days yoksa, sym_on/hosp_vis'ten üret
    if "delay_days" not in df2.columns:
        if "sym_on" in df2.columns and "hosp_vis" in df2.columns:
            df2["sym_on"] = pd.to_datetime(df2["sym_on"], errors="coerce")
            df2["hosp_vis"] = pd.to_datetime(df2["hosp_vis"], errors="coerce")
            df2["delay_days"] = (df2["hosp_vis"] - df2["sym_on"]).dt.days
            df2["delay_days"] = df2["delay_days"].clip(lower=0, upper=30)
            df2["delay_days"] = df2["delay_days"].fillna(0).astype(int)
            df2 = df2.drop(columns=["sym_on", "hosp_vis"], errors="ignore")
        else:
            df2["delay_days"] = 0

    # delay_days temizle
    df2["delay_days"] = pd.to_numeric(df2["delay_days"], errors="coerce").fillna(0)
    df2["delay_days"] = df2["delay_days"].clip(lower=0, upper=30).astype(int)

    return df2

if uploaded_file:
    try:
        data = pd.read_csv(uploaded_file)
        st.write("Yüklenen Veri (İlk 5 satır):")
        st.dataframe(data.head(), use_container_width=True)

        if st.button("Tüm Listeyi Tahmin Et"):
            data_proc = data.copy()

            # id varsa at
            if "id" in data_proc.columns:
                data_proc = data_proc.drop(columns=["id"], errors="ignore")

            # delay_days üret / normalize et
            data_proc = ensure_delay_days(data_proc)

            # Eksik zorunlu kolonlar var mı kontrol et (pipeline zaten hata verebilir ama burada daha okunaklı)
            required_cols = ["age", "gender", "country", "location", "vis_wuhan", "from_wuhan", "recov", "delay_days",
                             "symptom1", "symptom2", "symptom3", "symptom4", "symptom5", "symptom6"]
            missing = [c for c in required_cols if c not in data_proc.columns]
            if missing:
                st.warning(
                    "CSV'de bazı kolonlar eksik. Eksik kolonlar NaN ile doldurulacak:\n\n"
                    + ", ".join(missing)
                )
                for c in missing:
                    data_proc[c] = np.nan

            preds = pipe.predict(data_proc)

            # Olasılık varsa ekle
            risk_prob = None
            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(data_proc)
                if proba.shape[1] >= 2:
                    risk_prob = proba[:, 1]

            out = data.copy()
            out["Tahmin_Sonucu"] = preds
            out["Tahmin_Sonucu"] = out["Tahmin_Sonucu"].map({0: "İyileşme", 1: "Ölüm/Risk"})

            if risk_prob is not None:
                out["Risk_Olasiligi"] = (risk_prob * 100).round(2)

            st.success("✅ Tahminler tamamlandı!")
            st.dataframe(out, use_container_width=True)

            csv_bytes = out.to_csv(index=False).encode("utf-8")
            st.download_button("📤 Sonuçları İndir (CSV)", csv_bytes, "tahmin_sonuclari.csv")

    except Exception as e:
        st.error(f"Hata: {e}")
