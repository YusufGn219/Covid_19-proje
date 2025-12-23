import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import date

# -----------------------------------------------------------------------------
# SAYFA AYARLARI
# -----------------------------------------------------------------------------
st.set_page_config(page_title="COVID-19 Risk Tahmini", page_icon="🦠", layout="wide")

st.markdown("<h1 style='color:#d63031; font-size:42px;'>🦠 COVID-19 Ölüm Riski Tahmini</h1>", unsafe_allow_html=True)
st.markdown(
    "<div style='color:#636e72; font-size:16px;'>"
    "Hastanın semptom ve demografik bilgilerine göre ölüm riski analizi yapar."
    "</div>",
    unsafe_allow_html=True
)
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# PIPELINE YÜKLEME
# -----------------------------------------------------------------------------
@st.cache_resource
def load_pipeline():
    path = "covid_pipeline.pkl"
    if not os.path.exists(path):
        st.error(
            f"⚠️ '{path}' bulunamadı.\n\n"
            "Notebook'ta modeli eğitip covid_pipeline.pkl üretmelisin ve proje köküne koymalısın."
        )
        return None
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"⚠️ Model yüklenirken hata oluştu: {e}")
        return None

pipe = load_pipeline()
if pipe is None:
    st.stop()

# -----------------------------------------------------------------------------
# UI SEÇENEKLERİ (dataset ile uyumlu)
# -----------------------------------------------------------------------------
COUNTRIES = [
    'China', 'France', 'Japan', 'Malaysia', 'Nepal', 'Singapore', 'South Korea', 'Taiwan',
    'Thailand', 'USA', 'Vietnam', 'Australia', 'Canada', 'Cambodia', 'Sri Lanka', 'Germany',
    'Finland', 'UAE', 'Philippines', 'India', 'Italy', 'UK', 'Russia', 'Sweden', 'Spain',
    'Belgium', 'Other'
]
LOCATIONS = ['Wuhan', 'Beijing', 'Shanghai', 'Guangdong', 'Other']
GENDERS = ['male', 'female']
SYMPTOMS = [
    'fever', 'cough', 'sore throat', 'runny nose', 'dyspnea', 'pneumonia',
    'headache', 'vomiting', 'diarrhea', 'fatigue', 'chill', 'body pain', 'malaise'
]

# -----------------------------------------------------------------------------
# SIDEBAR: GİRİŞLER
# -----------------------------------------------------------------------------
st.sidebar.header("📝 Hasta Bilgileri")

country = st.sidebar.selectbox("Ülke (country)", COUNTRIES, index=COUNTRIES.index("Other") if "Other" in COUNTRIES else 0)
location = st.sidebar.selectbox("Bölge/Şehir (location)", LOCATIONS, index=LOCATIONS.index("Other") if "Other" in LOCATIONS else 0)
gender = st.sidebar.selectbox("Cinsiyet (gender)", GENDERS, index=0)
age = st.sidebar.slider("Yaş (age)", 0, 100, 35)

st.sidebar.markdown("---")
st.sidebar.markdown("##### 📅 Tarih Bilgileri (sym_on, hosp_vis)")
use_dates = st.sidebar.checkbox("Tarih gireceğim", value=False)

sym_on = None
hosp_vis = None
if use_dates:
    sym_on = st.sidebar.date_input("Semptom Başlangıç Tarihi (sym_on)", value=date(2020, 1, 1))
    hosp_vis = st.sidebar.date_input("Hastaneye Geliş Tarihi (hosp_vis)", value=date(2020, 1, 2))

st.sidebar.markdown("---")
st.sidebar.markdown("##### ✈️ Seyahat ve Geçmiş")
vis_wuhan = st.sidebar.radio("Wuhan'ı ziyaret etti mi? (vis_wuhan)", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")
from_wuhan = st.sidebar.radio("Wuhan'dan mı geldi? (from_wuhan)", [0, 1], format_func=lambda x: "Evet" if x == 1 else "Hayır")
recov = st.sidebar.radio("İyileşme durumu (recov)", [0, 1], index=0, format_func=lambda x: "İyileşti" if x == 1 else "Bilinmiyor/Hayır")

st.sidebar.markdown("---")
st.sidebar.markdown("##### 🤒 Semptomlar (symptom1..symptom6)")
symptom1 = st.sidebar.selectbox("Semptom 1 (symptom1)", [""] + SYMPTOMS)
symptom2 = st.sidebar.selectbox("Semptom 2 (symptom2)", [""] + SYMPTOMS)
symptom3 = st.sidebar.selectbox("Semptom 3 (symptom3)", [""] + SYMPTOMS)
symptom4 = st.sidebar.selectbox("Semptom 4 (symptom4)", [""] + SYMPTOMS)
symptom5 = st.sidebar.selectbox("Semptom 5 (symptom5)", [""] + SYMPTOMS)
symptom6 = st.sidebar.selectbox("Semptom 6 (symptom6)", [""] + SYMPTOMS)

# -----------------------------------------------------------------------------
# YARDIMCI: delay_days HESAPLA
# -----------------------------------------------------------------------------
def calc_delay_days(sym_on_val, hosp_vis_val) -> int:
    if sym_on_val is None or hosp_vis_val is None:
        return 0
    d1 = pd.to_datetime(sym_on_val)
    d2 = pd.to_datetime(hosp_vis_val)
    diff = (d2 - d1).days
    if diff is None:
        return 0
    return int(np.clip(diff, 0, 30))

# -----------------------------------------------------------------------------
# TEK SATIRLIK INPUT DF
# -----------------------------------------------------------------------------
def build_single_input_df() -> pd.DataFrame:
    delay_days = calc_delay_days(sym_on, hosp_vis) if use_dates else 0

    row = {
        "location": location,
        "country": country,
        "gender": gender,
        "age": age,
        "vis_wuhan": vis_wuhan,
        "from_wuhan": from_wuhan,
        "recov": recov,
        "symptom1": symptom1 if symptom1 else np.nan,
        "symptom2": symptom2 if symptom2 else np.nan,
        "symptom3": symptom3 if symptom3 else np.nan,
        "symptom4": symptom4 if symptom4 else np.nan,
        "symptom5": symptom5 if symptom5 else np.nan,
        "symptom6": symptom6 if symptom6 else np.nan,
        "delay_days": delay_days,
    }
    return pd.DataFrame([row])

# -----------------------------------------------------------------------------
# BAŞLIK: TEKİL TAHMİN
# -----------------------------------------------------------------------------
st.subheader("🔍 Tekil Tahmin")

col1, col2 = st.columns([1, 1], vertical_alignment="top")

with col1:
    st.markdown("**Girdi Özeti**")
    preview_df = build_single_input_df()
    st.dataframe(preview_df, use_container_width=True)

with col2:
    st.markdown("**Sonuç**")

    if st.button("▶︎ Riski Tahmin Et"):
        try:
            X = build_single_input_df()
            pred = int(pipe.predict(X)[0])

            proba_txt = ""
            if hasattr(pipe, "predict_proba"):
                p = pipe.predict_proba(X)[0]
                if len(p) >= 2:
                    proba_txt = f" (Risk olasılığı: **%{p[1]*100:.1f}**)"

            if pred == 1:
                st.error(f"⚠️ Tahmin: **Yüksek Risk / Ölüm (1)**{proba_txt}")
            else:
                st.success(f"✅ Tahmin: **Düşük Risk / Yaşam (0)**{proba_txt}")

        except Exception as e:
            st.error(f"Bir hata oluştu: {e}")

# -----------------------------------------------------------------------------
# TOPLU TAHMİN
# -----------------------------------------------------------------------------
st.markdown("<hr style='border:1px solid #eee;'>", unsafe_allow_html=True)
st.subheader("📂 Toplu Tahmin (CSV Yükleme)")

st.markdown(
    "<div style='color:#636e72; font-size:14px;'>"
    "CSV’de şu kolonlar olmalı: location, country, gender, age, vis_wuhan, from_wuhan, recov, "
    "symptom1..symptom6 ve (delay_days) veya (sym_on + hosp_vis). "
    "id varsa otomatik atılır."
    "</div>",
    unsafe_allow_html=True
)

uploaded = st.file_uploader("CSV seç", type=["csv"])

def prepare_batch_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    # id at
    out = out.drop(columns=["id"], errors="ignore")

    # delay_days yoksa sym_on/hosp_vis'ten üret
    if "delay_days" not in out.columns:
        if "sym_on" in out.columns and "hosp_vis" in out.columns:
            out["sym_on"] = pd.to_datetime(out["sym_on"], errors="coerce")
            out["hosp_vis"] = pd.to_datetime(out["hosp_vis"], errors="coerce")
            out["delay_days"] = (out["hosp_vis"] - out["sym_on"]).dt.days
            out["delay_days"] = out["delay_days"].clip(lower=0, upper=30).fillna(0).astype(int)
            out = out.drop(columns=["sym_on", "hosp_vis"], errors="ignore")
        else:
            out["delay_days"] = 0

    # delay_days temizle
    out["delay_days"] = pd.to_numeric(out["delay_days"], errors="coerce").fillna(0)
    out["delay_days"] = out["delay_days"].clip(lower=0, upper=30).astype(int)

    # Eksik kolonları ekle (pipeline patlamasın)
    required_cols = [
        "location", "country", "gender", "age", "vis_wuhan", "from_wuhan", "recov",
        "symptom1", "symptom2", "symptom3", "symptom4", "symptom5", "symptom6",
        "delay_days"
    ]
    for c in required_cols:
        if c not in out.columns:
            out[c] = np.nan

    return out[required_cols]

if uploaded:
    try:
        raw = pd.read_csv(uploaded)
        st.write("Yüklenen veri (ilk 5):")
        st.dataframe(raw.head(), use_container_width=True)

        if st.button("Tüm Listeyi Tahmin Et"):
            Xb = prepare_batch_df(raw)

            preds = pipe.predict(Xb)
            result = raw.copy()
            result["Tahmin_Sonucu"] = preds
            result["Tahmin_Sonucu"] = result["Tahmin_Sonucu"].map({0: "Yaşam", 1: "Ölüm/Risk"})

            if hasattr(pipe, "predict_proba"):
                proba = pipe.predict_proba(Xb)
                if proba.shape[1] >= 2:
                    result["Risk_Olasiligi"] = (proba[:, 1] * 100).round(2)

            st.success("✅ Tahminler tamamlandı.")
            st.dataframe(result, use_container_width=True)

            st.download_button(
                "📥 Sonuçları indir (CSV)",
                result.to_csv(index=False).encode("utf-8"),
                "tahmin_sonuclari.csv"
            )

    except Exception as e:
        st.error(f"Hata: {e}")
