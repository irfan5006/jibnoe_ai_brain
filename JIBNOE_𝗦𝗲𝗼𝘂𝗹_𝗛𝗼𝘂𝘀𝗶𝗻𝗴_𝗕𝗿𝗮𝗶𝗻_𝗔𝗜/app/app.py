import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.compose import ColumnTransformer

# ── Compatibility patch ──────────────────────────────────────────────────────
if not hasattr(ColumnTransformer, 'force_int_remainder_cols'):
    ColumnTransformer.force_int_remainder_cols = property(lambda self: False)

# ── Path configurations ──────────────────────────────────────────────────────
MODEL_DIR = 'e:/Machine_Learning/k_housing_brain_ai/models'
DATA_PATH = 'e:/Machine_Learning/k_housing_brain_ai/data/processed_data/processed_data.csv'

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="집뇌 JIBNOE — Seoul Housing Brain AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=DM+Serif+Display:ital@0;1&display=swap');

:root {
    --bg:          #0D1117;
    --bg2:         #161B24;
    --bg3:         #1E2533;
    --border:      rgba(255,255,255,0.07);
    --border2:     rgba(255,255,255,0.12);
    --mint:        #4ECDA4;
    --mint2:       #2BB585;
    --mint-glow:   rgba(78,205,164,0.18);
    --coral:       #FF7C5C;
    --gold:        #F5C842;
    --text:        #E8EDF5;
    --text2:       #8A95A8;
    --text3:       #556070;
    --radius:      16px;
    --radius-sm:   10px;
    --tr:          all 0.3s cubic-bezier(0.4,0,0.2,1);
}
* { font-family: 'Noto Sans KR', sans-serif !important; }
.stApp {
    background: var(--bg) !important;
    background-image:
        radial-gradient(ellipse 70% 50% at 15% 0%,  rgba(78,205,164,0.08) 0%, transparent 60%),
        radial-gradient(ellipse 50% 40% at 90% 100%, rgba(255,124,92,0.07) 0%, transparent 60%) !important;
}
#MainMenu, footer, header { visibility: hidden !important; }
.block-container { max-width: 1080px !important; padding: 0 2rem 5rem !important; }

.jib-nav {
    background: rgba(13,17,23,0.92); backdrop-filter: blur(24px);
    border-bottom: 1px solid var(--border2); padding: 14px 36px;
    display: flex; align-items: center; justify-content: space-between;
    margin: -1rem -2rem 2.5rem -2rem; position: sticky; top: 0; z-index: 999;
}
.jib-logo-wrap { display: flex; align-items: center; gap: 14px; }
.jib-brain-icon {
    width: 44px; height: 44px; border-radius: 13px;
    background: linear-gradient(135deg, var(--mint2), #1A8C65);
    display: flex; align-items: center; justify-content: center; font-size: 22px;
    box-shadow: 0 0 20px var(--mint-glow), 0 4px 12px rgba(0,0,0,0.4);
}
.jib-logo-name { font-family: 'DM Serif Display', serif !important; font-size: 26px; color: var(--text); letter-spacing: -0.5px; line-height: 1; }
.jib-logo-name span { color: var(--mint); }
.jib-logo-ko { font-size: 10px; color: var(--text3); letter-spacing: 2.5px; text-transform: uppercase; margin-top: 2px; }
.jib-nav-right { display: flex; align-items: center; gap: 10px; }
.jib-badge-green { background: rgba(78,205,164,0.12); border: 1px solid rgba(78,205,164,0.3); color: var(--mint); font-size: 11px; font-weight: 700; padding: 6px 14px; border-radius: 20px; }
.jib-dev-badge { background: rgba(245,200,66,0.10); border: 1px solid rgba(245,200,66,0.25); color: var(--gold); font-size: 11px; font-weight: 600; padding: 6px 14px; border-radius: 20px; }

.jib-hero {
    background: var(--bg2); border: 1px solid var(--border2); border-radius: 24px;
    padding: 52px 56px; margin-bottom: 28px; position: relative; overflow: hidden;
}
.jib-hero::before {
    content: ''; position: absolute; top: -120px; right: -120px;
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(78,205,164,0.10) 0%, transparent 65%);
    border-radius: 50%; animation: pulse1 8s ease-in-out infinite alternate; pointer-events: none;
}
.jib-hero::after {
    content: ''; position: absolute; bottom: -80px; left: 30%;
    width: 360px; height: 360px;
    background: radial-gradient(circle, rgba(255,124,92,0.08) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
}
@keyframes pulse1 { from{transform:scale(1);} to{transform:scale(1.25) rotate(15deg);} }
.jib-hero-eyebrow {
    display: inline-flex; align-items: center; gap: 8px;
    background: rgba(78,205,164,0.10); border: 1px solid rgba(78,205,164,0.25);
    color: var(--mint); font-size: 11px; font-weight: 700; letter-spacing: 2px;
    text-transform: uppercase; padding: 6px 16px; border-radius: 20px; margin-bottom: 20px;
}
.jib-hero-title { font-family: 'DM Serif Display', serif !important; font-size: 46px; line-height: 1.15; color: var(--text); margin-bottom: 14px; font-weight: 400; }
.jib-hero-title .accent { color: var(--mint); }
.jib-hero-title .italic { font-style: italic; color: var(--coral); }
.jib-hero-desc { font-size: 14px; color: var(--text2); line-height: 1.8; max-width: 540px; margin-bottom: 36px; }
.jib-stats { display: flex; gap: 0; position: relative; z-index: 1; flex-wrap: wrap; }
.jib-stat { padding: 16px 28px; border-right: 1px solid var(--border); position: relative; }
.jib-stat:first-child { padding-left: 0; }
.jib-stat:last-child  { border-right: none; }
.jib-stat-num { font-family: 'DM Serif Display', serif !important; font-size: 30px; font-weight: 400; line-height: 1; color: var(--mint); }
.jib-stat-lbl { font-size: 11px; color: var(--text3); margin-top: 4px; }
.jib-stat-badge { position: absolute; top: 14px; right: 10px; background: rgba(78,205,164,0.12); color: var(--mint); font-size: 9px; font-weight: 800; padding: 2px 7px; border-radius: 10px; }

.acc-wrap { background: var(--bg2); border: 1px solid var(--border2); border-radius: var(--radius); padding: 20px 24px; display: flex; align-items: center; gap: 20px; margin-bottom: 24px; }
.acc-circle { width: 72px; height: 72px; border-radius: 50%; background: conic-gradient(var(--mint) 0% 90%, var(--bg3) 90% 100%); display: flex; align-items: center; justify-content: center; position: relative; flex-shrink: 0; box-shadow: 0 0 24px var(--mint-glow); }
.acc-circle::before { content: ''; position: absolute; inset: 8px; background: var(--bg2); border-radius: 50%; }
.acc-pct { position: relative; z-index: 1; font-size: 14px; font-weight: 800; color: var(--mint); }
.acc-info-title { font-size: 14px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.acc-info-body  { font-size: 12px; color: var(--text3); line-height: 1.7; }
.acc-tags { display: flex; gap: 6px; margin-top: 8px; flex-wrap: wrap; }

.jib-info-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 14px; margin-bottom: 24px; }
.jib-info-card { background: var(--bg2); border: 1px solid var(--border); border-radius: var(--radius); padding: 20px 22px; display: flex; gap: 14px; align-items: flex-start; transition: var(--tr); }
.jib-info-card:hover { border-color: var(--mint); transform: translateY(-3px); }
.jib-info-icon { font-size: 26px; flex-shrink: 0; line-height: 1; margin-top: 2px; }
.jib-info-title { font-size: 13px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.jib-info-body  { font-size: 12px; color: var(--text3); line-height: 1.6; }

.jib-card { background: var(--bg2); border: 1px solid var(--border2); border-radius: var(--radius); padding: 32px 36px; margin-bottom: 20px; position: relative; overflow: hidden; }
.jib-card::before { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px; background: linear-gradient(90deg, var(--mint2), var(--mint), var(--coral)); border-radius: var(--radius) var(--radius) 0 0; }
.jib-card-title { font-size: 19px; font-weight: 700; color: var(--text); margin-bottom: 4px; }
.jib-card-sub   { font-size: 13px; color: var(--text3); margin-bottom: 24px; padding-bottom: 20px; border-bottom: 1px solid var(--border); }
.jib-section-chip { display: inline-flex; align-items: center; gap: 6px; background: rgba(78,205,164,0.08); border: 1px solid rgba(78,205,164,0.20); color: var(--mint); font-size: 10px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; padding: 5px 14px; border-radius: 20px; margin-bottom: 14px; }
.jib-hint { background: rgba(255,255,255,0.03); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 7px 13px; font-size: 11px; color: var(--text3); margin-top: -6px; margin-bottom: 6px; }

div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label { font-size: 13px !important; font-weight: 600 !important; color: var(--text2) !important; }
div[data-testid="stNumberInput"] input { background: var(--bg3) !important; border: 1.5px solid var(--border2) !important; border-radius: var(--radius-sm) !important; color: var(--text) !important; font-size: 15px !important; padding: 10px 14px !important; }
div[data-testid="stNumberInput"] input:focus { border-color: var(--mint) !important; box-shadow: 0 0 0 3px var(--mint-glow) !important; }
div[data-testid="stSelectbox"] > div > div { background: var(--bg3) !important; border: 1.5px solid var(--border2) !important; border-radius: var(--radius-sm) !important; color: var(--text) !important; }
div[data-testid="stSelectbox"] > div > div:focus-within { border-color: var(--mint) !important; }

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, var(--mint2) 0%, #1E9E74 100%) !important;
    color: #061A10 !important; border: none !important; border-radius: 50px !important;
    font-size: 16px !important; font-weight: 800 !important; padding: 14px 40px !important;
    width: 100% !important; letter-spacing: 0.5px !important;
    box-shadow: 0 0 30px rgba(43,181,133,0.35), 0 6px 20px rgba(0,0,0,0.4) !important;
    transition: var(--tr) !important; margin-top: 10px !important;
}
div[data-testid="stButton"] > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 0 50px rgba(43,181,133,0.5), 0 12px 32px rgba(0,0,0,0.5) !important;
}

.result-wrap { margin-top: 28px; animation: popIn 0.55s cubic-bezier(0.34,1.56,0.64,1) forwards; }
@keyframes popIn { from{opacity:0;transform:scale(0.94) translateY(16px);} to{opacity:1;transform:scale(1) translateY(0);} }
.result-glow-ring { background: linear-gradient(135deg, var(--mint2), var(--mint), var(--coral)); border-radius: calc(var(--radius) + 2px); padding: 2px; box-shadow: 0 0 60px rgba(78,205,164,0.25), 0 20px 60px rgba(0,0,0,0.5); }
.result-inner { background: var(--bg2); border-radius: var(--radius); padding: 36px 40px; }
.result-eyebrow { display: flex; align-items: center; gap: 8px; font-size: 11px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; color: var(--mint); margin-bottom: 12px; }
.result-eyebrow .dot { width: 6px; height: 6px; border-radius: 50%; background: var(--mint); box-shadow: 0 0 8px var(--mint); animation: blink 1.4s ease-in-out infinite; }
@keyframes blink { 0%,100%{opacity:1;} 50%{opacity:0.3;} }
.result-price { font-family: 'DM Serif Display', serif !important; font-size: 58px; font-weight: 400; line-height: 1; color: var(--text); margin-bottom: 6px; letter-spacing: -2px; }
.result-price .currency { color: var(--mint); font-size: 36px; }
.result-sub { font-size: 14px; color: var(--text3); margin-bottom: 28px; }
.result-divider { height: 1px; background: var(--border2); margin-bottom: 24px; position: relative; }
.result-divider::after { content: '✦'; position: absolute; top: 50%; left: 50%; transform: translate(-50%,-50%); background: var(--bg2); padding: 0 10px; color: var(--text3); font-size: 10px; }
.result-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; margin-bottom: 24px; }
.result-cell { background: var(--bg3); border: 1px solid var(--border); border-radius: var(--radius-sm); padding: 14px 16px; transition: var(--tr); }
.result-cell:hover { border-color: var(--mint); }
.result-cell-lbl { font-size: 10px; color: var(--text3); text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px; font-weight: 600; }
.result-cell-val { font-size: 15px; font-weight: 700; color: var(--text); }
.result-tags { display: flex; gap: 8px; flex-wrap: wrap; }
.rtag { padding: 6px 14px; border-radius: 20px; font-size: 11px; font-weight: 700; }
.rtag-mint  { background: rgba(78,205,164,0.12); color: var(--mint);  border: 1px solid rgba(78,205,164,0.25); }
.rtag-coral { background: rgba(255,124,92,0.12);  color: var(--coral); border: 1px solid rgba(255,124,92,0.25); }
.rtag-gold  { background: rgba(245,200,66,0.10);  color: var(--gold);  border: 1px solid rgba(245,200,66,0.22); }
.rtag-gray  { background: rgba(255,255,255,0.05); color: var(--text2); border: 1px solid var(--border); }

.jib-error { background: rgba(255,124,92,0.08); border: 1px solid rgba(255,124,92,0.3); border-left: 4px solid var(--coral); border-radius: var(--radius-sm); padding: 16px 20px; font-size: 13px; color: #FF9B84; margin-top: 20px; }

.jib-footer { text-align: center; padding: 28px 20px; border-top: 1px solid var(--border); margin-top: 48px; }
.jib-footer-logo { font-family: 'DM Serif Display', serif !important; font-size: 20px; color: var(--text2); margin-bottom: 8px; }
.jib-footer-logo span { color: var(--mint); }
.jib-footer-dev  { font-size: 13px; font-weight: 600; color: var(--gold); margin-bottom: 6px; }
.jib-footer-body { font-size: 11px; color: var(--text3); line-height: 1.8; }

@media (max-width: 768px) {
    .jib-hero { padding: 28px 24px; }
    .jib-hero-title { font-size: 30px; }
    .jib-info-row { grid-template-columns: 1fr; }
    .result-grid  { grid-template-columns: 1fr 1fr; }
    .result-price { font-size: 40px; }
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    pipeline = joblib.load(f'{MODEL_DIR}/full_preprocess_pipeline.pkl')
    model    = joblib.load(f'{MODEL_DIR}/random_forest_model.pkl')
    return pipeline, model

@st.cache_data
def load_dropdown_data():
    try:
        df     = pd.read_csv(DATA_PATH)
        dongs  = sorted(df['legal_dong_name'].dropna().unique().tolist())
        usages = sorted(df['building_usage_en'].dropna().unique().tolist())
        return dongs, usages
    except Exception as e:
        st.warning(f"CSV not found ({e}). Using demo values.")
        return (
            ["nonhyeon","apgujeong","itaewon","hongdae","seongsu","mapo","jongno","songpa"],
            ["Apartment","Officetel","Villa","Single House","Commercial"],
        )


# ── NAVBAR ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="jib-nav">
  <div class="jib-logo-wrap">
    <div class="jib-brain-icon">🧠</div>
    <div>
      <div class="jib-logo-name">JIB<span>NOE</span> 집뇌</div>
      <div class="jib-logo-ko">Seoul Housing Brain AI</div>
    </div>
  </div>
  <div class="jib-nav-right">
    <div class="jib-badge-green">⚡ Model Live</div>
    <div class="jib-dev-badge">👨‍💻 Muhammad Irfan</div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── HERO ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="jib-hero">
  <div class="jib-hero-eyebrow">🧠 집뇌 · Seoul Housing Brain AI</div>
  <div class="jib-hero-title">
    The Brain That<br/>
    <span class="italic">Knows</span> Your Home's<br/>
    <span class="accent">True Value</span>
  </div>
  <div class="jib-hero-desc">
    서울 부동산 시세를 직접 훈련한 Random Forest 모델로 정확히 예측합니다.<br/>
    Enter property details and let the AI brain calculate the market price instantly.
  </div>
  <div class="jib-stats">
    <div class="jib-stat">
      <div class="jib-stat-num">90%</div>
      <div class="jib-stat-lbl">Model Accuracy</div>
      <div class="jib-stat-badge">TRAINED</div>
    </div>
    <div class="jib-stat">
      <div class="jib-stat-num">RF</div>
      <div class="jib-stat-lbl">Random Forest</div>
    </div>
    <div class="jib-stat">
      <div class="jib-stat-num">7</div>
      <div class="jib-stat-lbl">Input Features</div>
    </div>
    <div class="jib-stat">
      <div class="jib-stat-num">Seoul</div>
      <div class="jib-stat-lbl">서울 전 지역</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── ACCURACY WIDGET ───────────────────────────────────────────────────────────
st.markdown("""
<div class="acc-wrap">
  <div class="acc-circle"><div class="acc-pct">90%</div></div>
  <div>
    <div class="acc-info-title">🧠 Custom-Trained Random Forest — by Muhammad Irfan</div>
    <div class="acc-info-body">
      Trained from scratch on Seoul real estate transaction data — not a pretrained model.
      Full pipeline includes custom preprocessing, feature engineering, and hyperparameter tuning.
    </div>
    <div class="acc-tags">
      <span class="rtag rtag-mint">✅ 90% Accuracy</span>
      <span class="rtag rtag-gold">🔬 Custom Trained</span>
      <span class="rtag rtag-gray">🌳 Random Forest</span>
      <span class="rtag rtag-coral">🏙 Seoul Dataset</span>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── INFO CARDS ────────────────────────────────────────────────────────────────
st.markdown("""
<div class="jib-info-row">
  <div class="jib-info-card">
    <div class="jib-info-icon">📐</div>
    <div>
      <div class="jib-info-title">면적 기반 분석</div>
      <div class="jib-info-body">Building & land area are the top price drivers learned by the model.</div>
    </div>
  </div>
  <div class="jib-info-card">
    <div class="jib-info-icon">📍</div>
    <div>
      <div class="jib-info-title">위치 프리미엄 인코딩</div>
      <div class="jib-info-body">Dong name encodes neighborhood premium from Seoul transaction history.</div>
    </div>
  </div>
  <div class="jib-info-card">
    <div class="jib-info-icon">⚡</div>
    <div>
      <div class="jib-info-title">즉시 예측 Instant Result</div>
      <div class="jib-info-body">Millisecond results via serialized sklearn preprocessing + RF pipeline.</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ── FORM ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="jib-card">
  <div class="jib-card-title">📊 가격 예측 · Price Prediction</div>
  <div class="jib-card-sub">Fill in all fields and press Predict — the brain does the rest.</div>
</div>
""", unsafe_allow_html=True)

pipeline, model = load_models()
dongs, usages   = load_dropdown_data()

st.markdown('<div class="jib-section-chip">🏗 Physical Attributes · 물리적 속성</div>', unsafe_allow_html=True)
col1, col2 = st.columns(2, gap="large")

with col1:
    building_area_m2 = st.number_input("🏢 Building Area (m²) · 건물 면적", min_value=10.0, max_value=500.0, value=84.0, step=1.0)
    st.markdown('<div class="jib-hint">💡 Typical Seoul apartments: 59 – 115 m²</div>', unsafe_allow_html=True)
    land_area_m2 = st.number_input("🌿 Land Area (m²) · 토지 면적", min_value=0.0, max_value=500.0, value=40.0, step=1.0)
    floor = st.number_input("🏢 Floor · 층수", min_value=-5, max_value=100, value=5, step=1)
    st.markdown('<div class="jib-hint">💡 Floors 15F+ add ~5–12% premium in Seoul</div>', unsafe_allow_html=True)

with col2:
    land_building_ratio = st.number_input("📐 Land to Building Ratio · 토지건물비율", min_value=0.0, max_value=5.0, value=0.5, step=0.1)
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="jib-section-chip">📍 Location & Context · 위치 및 맥락</div>', unsafe_allow_html=True)

    try:    dong_index = dongs.index('nonhyeon') if 'nonhyeon' in dongs else 0
    except: dong_index = 0
    legal_dong_name = st.selectbox("📍 Legal Dong Name · 법정동 이름", dongs, index=dong_index)

    try:    usage_index = usages.index('Apartment') if 'Apartment' in usages else 0
    except: usage_index = 0
    building_usage_en = st.selectbox("🏠 Building Usage · 건물 용도", usages, index=usage_index)

    month = st.selectbox(
        "📅 Month of Sale · 판매 월", list(range(1, 13)), index=5,
        format_func=lambda m: ["January 1월","February 2월","March 3월","April 4월",
            "May 5월","June 6월","July 7월","August 8월",
            "September 9월","October 10월","November 11월","December 12월"][m-1]
    )


# ── PREDICT ───────────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
predict_clicked = st.button("🧠 뇌를 깨워라 · Activate the Brain · Predict Price", use_container_width=True)

if predict_clicked:
    with st.spinner("🧠 집뇌가 분석 중입니다… JIBNOE is calculating…"):
        input_data = pd.DataFrame({
            'building_area_m2':    [building_area_m2],
            'land_area_m2':        [land_area_m2],
            'floor':               [floor],
            'month':               [month],
            'legal_dong_name':     [legal_dong_name],
            'building_usage_en':   [building_usage_en],
            'land_building_ratio': [land_building_ratio],
        })
        try:
            prediction_units = model.predict(input_data)[0]
            actual_krw       = prediction_units * 10_000

            if actual_krw >= 1_000_000_000:
                price_main, price_unit, unit_label = f"{actual_krw/1_000_000_000:.2f}", "B KRW", "십억원 Billion KRW"
            elif actual_krw >= 1_000_000:
                price_main, price_unit, unit_label = f"{actual_krw/1_000_000:.0f}", "M KRW", "백만원 Million KRW"
            else:
                price_main, price_unit, unit_label = f"{actual_krw:,.0f}", "KRW", "원 KRW"

            raw_krw_fmt = f"₩ {actual_krw:,.0f}"
            manwon_fmt  = f"₩ {prediction_units:,.1f} 만원"
            per_m2_fmt  = f"₩ {(actual_krw/building_area_m2)/1_000_000:.1f}M / m²" if building_area_m2 > 0 else "N/A"

            t1 = '<span class="rtag rtag-coral">🔥 하이라이즈 High-rise</span>'   if floor >= 15 else \
                 '<span class="rtag rtag-mint">📈 미드하이 Mid-high Floor</span>'  if floor >= 8  else \
                 '<span class="rtag rtag-gray">🏢 Standard Floor</span>'
            t2 = '<span class="rtag rtag-mint">✅ 적정 면적 Typical Size</span>'   if 50 <= building_area_m2 <= 130 else \
                 '<span class="rtag rtag-coral">📐 비전형 Atypical Size</span>'
            t3 = '<span class="rtag rtag-gold">💎 프리미엄 Premium Tier</span>'   if actual_krw >= 1_000_000_000 else \
                 '<span class="rtag rtag-mint">🏠 Mid-Market</span>'               if actual_krw >= 500_000_000   else \
                 '<span class="rtag rtag-gray">💼 Entry Market</span>'

            st.markdown(f"""
            <div class="result-wrap">
              <div class="result-glow-ring">
                <div class="result-inner">
                  <div class="result-eyebrow"><div class="dot"></div>예측 완료 · Prediction Complete · JIBNOE 집뇌</div>
                  <div class="result-price"><span class="currency">₩</span>{price_main} {price_unit}</div>
                  <div class="result-sub">{raw_krw_fmt} &nbsp;·&nbsp; {unit_label}</div>
                  <div class="result-divider"></div>
                  <div class="result-grid">
                    <div class="result-cell"><div class="result-cell-lbl">만원 단위 Manwon</div><div class="result-cell-val">{manwon_fmt}</div></div>
                    <div class="result-cell"><div class="result-cell-lbl">단가 Price / m²</div><div class="result-cell-val">{per_m2_fmt}</div></div>
                    <div class="result-cell"><div class="result-cell-lbl">위치 Location</div><div class="result-cell-val">📍 {legal_dong_name}</div></div>
                    <div class="result-cell"><div class="result-cell-lbl">건물 면적 Area</div><div class="result-cell-val">{building_area_m2:.0f} m²</div></div>
                    <div class="result-cell"><div class="result-cell-lbl">층수 Floor</div><div class="result-cell-val">{floor}F</div></div>
                    <div class="result-cell"><div class="result-cell-lbl">판매월 Month</div><div class="result-cell-val">Month {month}</div></div>
                  </div>
                  <div class="result-tags">{t1}{t2}{t3}<span class="rtag rtag-gray">🧠 JIBNOE 집뇌 · 90% Acc</span></div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        except Exception as e:
            st.markdown(f"""
            <div class="jib-error">
              ⚠️ <strong>예측 오류 · Prediction Error</strong><br/><br/>
              {e}<br/><br/>
              Check model file paths and ensure input data matches the training format.
            </div>
            """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="jib-footer">
  <div class="jib-footer-logo">JIB<span>NOE</span> 집뇌</div>
  <div class="jib-footer-dev">👨‍💻 Developed by Muhammad Irfan</div>
  <div class="jib-footer-body">
    Seoul Housing Brain AI · Custom-Trained Random Forest · 90% Accuracy<br/>
    서울 부동산 시세 예측 AI · 직접 훈련한 머신러닝 모델 기반<br/>
    © 2025 JIBNOE 집뇌 · All rights reserved
  </div>
</div>
""", unsafe_allow_html=True)