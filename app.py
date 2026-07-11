import streamlit as st
import numpy as np
import pickle
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import base64

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #0D1B4B 0%, #1a237e 50%, #0891B2 100%);
        padding: 30px 40px; border-radius: 18px; margin-bottom: 28px;
        box-shadow: 0 8px 32px rgba(13,27,75,0.25);
    }
    .main-header h1 { color: white; font-size: 2.1rem; font-weight: 700; margin: 0; }
    .main-header p  { color: #90CAF9; font-size: 1rem; margin: 6px 0 0; }
    .main-header small { color: #78909C; font-size: 0.82rem; }

    .stat-card {
        background: white; border-radius: 14px; padding: 22px 18px;
        box-shadow: 0 2px 16px rgba(0,0,0,0.08); border-left: 5px solid #0891B2;
        text-align: center;
    }
    .stat-card .val { font-size: 2rem; font-weight: 700; color: #0D1B4B; }
    .stat-card .lbl { font-size: 0.82rem; color: #607D8B; margin-top: 4px; }

    .result-high {
        background: linear-gradient(135deg,#FFF5F5,#FED7D7);
        border-left: 6px solid #E53E3E; border-radius: 14px; padding: 22px;
        box-shadow: 0 4px 16px rgba(229,62,62,0.12);
    }
    .result-moderate {
        background: linear-gradient(135deg,#FFFBEB,#FEF3C7);
        border-left: 6px solid #D97706; border-radius: 14px; padding: 22px;
        box-shadow: 0 4px 16px rgba(217,119,6,0.12);
    }
    .result-low {
        background: linear-gradient(135deg,#F0FFF4,#C6F6D5);
        border-left: 6px solid #38A169; border-radius: 14px; padding: 22px;
        box-shadow: 0 4px 16px rgba(56,161,105,0.12);
    }
    .result-high h3    { color: #C53030; }
    .result-moderate h3 { color: #92400E; }
    .result-low h3     { color: #276749; }

    .section-title {
        font-size: 1.1rem; font-weight: 700; color: #0D1B4B;
        border-bottom: 2px solid #0891B2; padding-bottom: 6px; margin: 18px 0 14px;
    }
    .history-card {
        background: white; border-radius: 10px; padding: 14px 18px;
        margin-bottom: 10px; box-shadow: 0 1px 8px rgba(0,0,0,0.07);
        border-left: 4px solid #0891B2;
    }
    .badge-high     { background:#FED7D7; color:#C53030; padding:3px 10px;
        border-radius:20px; font-size:.78rem; font-weight:600; }
    .badge-moderate { background:#FEF3C7; color:#92400E; padding:3px 10px;
        border-radius:20px; font-size:.78rem; font-weight:600; }
    .badge-low      { background:#C6F6D5; color:#276749; padding:3px 10px;
        border-radius:20px; font-size:.78rem; font-weight:600; }

    .stButton > button {
        background: linear-gradient(135deg,#0D1B4B,#0891B2);
        color: white; border: none; border-radius: 10px; padding: 12px 28px;
        font-size: 1rem; font-weight: 600;
        box-shadow: 0 4px 14px rgba(8,145,178,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state.history = []

# ── Load Models ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    dm = pickle.load(open('diabetes_model.pkl',  'rb'))
    ds = pickle.load(open('diabetes_scaler.pkl', 'rb'))
    hm = pickle.load(open('heart_model.pkl',     'rb'))
    hs = pickle.load(open('heart_scaler.pkl',    'rb'))
    return dm, ds, hm, hs

diabetes_model, diabetes_scaler, heart_model, heart_scaler = load_models()

# ── Risk Logic ────────────────────────────────────────────────────────────────
def get_risk(prob):
    score = round(prob * 100, 1)
    if score < 30:
        return score, "LOW RISK",      "low",      "✅", "#38A169"
    elif score < 60:
        return score, "MODERATE RISK", "moderate", "⚠️", "#D97706"
    else:
        return score, "HIGH RISK",     "high",     "🚨", "#E53E3E"

# ── Plotly Gauge ──────────────────────────────────────────────────────────────
def make_gauge(score, color, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={'text': title, 'font': {'size': 14, 'color': '#0D1B4B'}},
        number={'suffix': '/100', 'font': {'size': 28, 'color': color}},
        gauge={
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "#555"},
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "#E2E8F0",
            'steps': [
                {'range': [0,  30], 'color': '#C6F6D5'},
                {'range': [30, 60], 'color': '#FEF3C7'},
                {'range': [60,100], 'color': '#FED7D7'},
            ],
            'threshold': {
                'line': {'color': color, 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    fig.update_layout(
        height=260, margin=dict(t=60, b=20, l=30, r=30),
        paper_bgcolor='#FAFAFA', font={'family': 'Inter'}
    )
    return fig

# ── Plotly Feature Bar ────────────────────────────────────────────────────────
def make_feature_bar(names, values, title, color='#0891B2'):
    numeric_names  = []
    numeric_values = []
    for n, v in zip(names, values):
        try:
            numeric_values.append(float(v))
            numeric_names.append(n)
        except (ValueError, TypeError):
            pass

    fig = go.Figure(go.Bar(
        x=numeric_values,
        y=numeric_names,
        orientation='h',
        marker_color=color,
        marker_line_color='white',
        marker_line_width=0.5,
        text=[str(v) for v in numeric_values],
        textposition='outside',
    ))
    fig.update_layout(
        title={'text': title, 'font': {'size': 13, 'color': '#0D1B4B'}},
        height=max(280, len(numeric_names) * 32),
        margin=dict(t=50, b=20, l=10, r=60),
        paper_bgcolor='#FAFAFA',
        plot_bgcolor='#FAFAFA',
        font={'family': 'Inter', 'size': 11},
        xaxis={'showgrid': True, 'gridcolor': '#E2E8F0'},
        yaxis={'showgrid': False},
    )
    return fig

# ── Plotly Risk Position Bar ──────────────────────────────────────────────────
def make_risk_bar(score, disease):
    fig = go.Figure()
    # Background zones
    for x0, x1, fc, label in [(0,30,'#C6F6D5','Low (0–30)'),
                                (30,60,'#FEF3C7','Moderate (30–60)'),
                                (60,100,'#FED7D7','High (60–100)')]:
        fig.add_shape(type='rect', x0=x0, x1=x1, y0=0, y1=1,
                      fillcolor=fc, line_width=0, layer='below')
        fig.add_annotation(x=(x0+x1)/2, y=0.5, text=label,
                           showarrow=False, font={'size':10,'color':'#555'})
    # Score marker
    color = '#38A169' if score<30 else ('#D97706' if score<60 else '#E53E3E')
    fig.add_shape(type='line', x0=score, x1=score, y0=0, y1=1,
                  line=dict(color=color, width=3, dash='dash'))
    fig.add_annotation(x=score, y=1.15, text=f'<b>{score}</b>',
                       showarrow=False, font={'size':14,'color':color})
    fig.update_layout(
        title={'text': f'{disease} — Risk Score Position',
               'font': {'size': 13, 'color': '#0D1B4B'}},
        xaxis={'range':[0,100],'showgrid':False,
               'tickvals':[0,30,60,100],'ticktext':['0','30','60','100']},
        yaxis={'visible':False,'range':[0,1.3]},
        height=150, margin=dict(t=50,b=20,l=20,r=20),
        paper_bgcolor='#FAFAFA', plot_bgcolor='#FAFAFA',
        font={'family':'Inter'}
    )
    return fig

# ── History Trend Chart ───────────────────────────────────────────────────────
def make_trend(history):
    df = pd.DataFrame(history)
    fig = go.Figure()
    for disease, color in [("Diabetes","#0891B2"),("Heart Disease","#E53E3E")]:
        sub = df[df['disease']==disease].reset_index(drop=True)
        if not sub.empty:
            fig.add_trace(go.Scatter(
                x=list(range(1, len(sub)+1)), y=sub['score'],
                mode='lines+markers', name=disease,
                line=dict(color=color, width=2),
                marker=dict(size=7)
            ))
    fig.add_hrect(y0=0,  y1=30,  fillcolor='#C6F6D5', opacity=0.2, line_width=0)
    fig.add_hrect(y0=30, y1=60,  fillcolor='#FEF3C7', opacity=0.2, line_width=0)
    fig.add_hrect(y0=60, y1=100, fillcolor='#FED7D7', opacity=0.2, line_width=0)
    fig.update_layout(
        title={'text':'📈 Risk Score Trend', 'font':{'size':13,'color':'#0D1B4B'}},
        xaxis={'title':'Prediction #'}, yaxis={'title':'Risk Score','range':[0,100]},
        height=300, margin=dict(t=50,b=40,l=50,r=20),
        paper_bgcolor='#FAFAFA', plot_bgcolor='#FAFAFA',
        font={'family':'Inter'}, legend={'orientation':'h','y':-0.2}
    )
    return fig

# ── PDF Report ────────────────────────────────────────────────────────────────
def generate_report(disease, score, level, prediction, inputs_dict, recs):
    lines = [
        "="*60,
        "     AI-POWERED HEALTH RISK PREDICTION REPORT",
        "     IIIT Senapati, Manipur — Pritesh Raj (230103035)",
        "="*60,
        f"\nDate & Time : {datetime.now().strftime('%d %B %Y, %I:%M %p')}",
        f"Disease     : {disease}",
        f"Prediction  : {prediction}",
        f"Risk Score  : {score}/100",
        f"Risk Level  : {level}",
        "\n--- PATIENT INPUT PARAMETERS ---",
    ]
    for k, v in inputs_dict.items():
        lines.append(f"  {k:<35}: {v}")
    lines.append("\n--- CLINICAL RECOMMENDATIONS ---")
    for i, r in enumerate(recs, 1):
        lines.append(f"  {i}. {r}")
    lines += ["\n"+"="*60,
              "DISCLAIMER: For educational purposes only.",
              "Always consult a qualified medical professional.",
              "="*60]
    return "\n".join(lines)

def download_btn(text, fname):
    b64 = base64.b64encode(text.encode()).decode()
    return (f'<a href="data:text/plain;base64,{b64}" download="{fname}" '
            f'style="text-decoration:none;">'
            f'<button style="background:linear-gradient(135deg,#0D1B4B,#0891B2);'
            f'color:white;border:none;border-radius:8px;padding:10px 22px;'
            f'font-size:0.9rem;font-weight:600;cursor:pointer;margin-top:10px;">'
            f'📄 Download Report</button></a>')

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🏥 AI Health Predictor")
    st.markdown("---")
    st.markdown("**📋 About**")
    st.info("Ensemble ML system predicting Diabetes & Heart Disease risk from clinical parameters.")
    st.markdown("**🤖 Best Models**")
    st.success("XGBoost — 76.62% (Diabetes)")
    st.success("XGBoost — 81.37% (Heart Disease)")
    st.markdown("---")
    st.markdown("**📊 Session Stats**")
    total = len(st.session_state.history)
    high  = sum(1 for h in st.session_state.history if 'HIGH' in h.get('level',''))
    st.metric("Total Predictions", total)
    st.metric("High Risk Cases",   high)
    st.markdown("---")
    st.markdown(
        "<small style='color:#90A4AE'>By Pritesh Raj | 230103035<br>"
        "Dept. CSE (AI & DS)<br>IIIT Senapati, Manipur</small>",
        unsafe_allow_html=True)

# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1>🏥 AI-Powered Dual Disease Prediction System</h1>
  <p>Diabetes & 10-Year Coronary Heart Disease Risk Using Ensemble Learning</p>
  <small>By Pritesh Raj | Roll No: 230103035 | IIIT Senapati, Manipur</small>
</div>
""", unsafe_allow_html=True)

# ── STAT CARDS ────────────────────────────────────────────────────────────────
c1,c2,c3,c4 = st.columns(4)
for col, val, lbl in zip(
    [c1,c2,c3,c4],
    ["768","4,240","7","81.37%"],
    ["Diabetes Records","Heart Disease Records","ML Models Compared","Best Accuracy"]
):
    col.markdown(
        f'<div class="stat-card"><div class="val">{val}</div>'
        f'<div class="lbl">{lbl}</div></div>',
        unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🩸 Diabetes Prediction",
    "❤️ Heart Disease Prediction",
    "🗂️ Prediction History",
    "ℹ️ About"
])

# ══════════════════════════════════════════════════════════════════
# TAB 1 — DIABETES
# ══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-title">🩸 Diabetes Risk Prediction</div>',
                unsafe_allow_html=True)
    st.caption("Enter the patient's clinical measurements. All fields are required.")

    with st.form("diabetes_form"):
        col1, col2 = st.columns(2)
        with col1:
            pregnancies    = st.number_input("🤰 Pregnancies",                0,  20,   1)
            glucose        = st.number_input("🩸 Glucose Level (mg/dL)",      0, 300, 120)
            blood_pressure = st.number_input("💓 Blood Pressure (mm Hg)",     0, 150,  70)
            skin_thickness = st.number_input("📏 Skin Thickness (mm)",        0, 100,  20)
        with col2:
            insulin = st.number_input("💉 Insulin Level (μIU/mL)",  0,   900,  80)
            bmi     = st.number_input("⚖️  BMI (kg/m²)",            0.0, 70.0, 25.0)
            dpf     = st.number_input("🧬 Diabetes Pedigree Function", 0.0, 3.0, 0.5)
            age     = st.number_input("🎂 Age (years)",              1,   120,  25)
        submitted = st.form_submit_button("🔍 Predict Diabetes Risk",
                                          use_container_width=True)

    if submitted:
        inp    = np.array([[pregnancies, glucose, blood_pressure,
                            skin_thickness, insulin, bmi, dpf, age]])
        scaled = diabetes_scaler.transform(inp)
        pred   = diabetes_model.predict(scaled)[0]
        prob   = diabetes_model.predict_proba(scaled)[0][1]
        score, level, css, icon, color = get_risk(prob)

        recs = (["Consult an endocrinologist immediately",
                 "Monitor blood glucose daily",
                 "Reduce sugar and refined carbohydrates",
                 "Exercise at least 30 minutes daily",
                 "Maintain healthy body weight",
                 "Consider HbA1c and OGTT tests"]
                if pred == 1 else
                ["Maintain current healthy lifestyle",
                 "Annual blood glucose check-up",
                 "Stay physically active (150 min/week)",
                 "Balanced diet with low glycaemic index",
                 "Avoid smoking and excessive alcohol"])

        inputs_dict = {
            "Pregnancies": pregnancies, "Glucose (mg/dL)": glucose,
            "Blood Pressure (mm Hg)": blood_pressure,
            "Skin Thickness (mm)": skin_thickness,
            "Insulin (μIU/mL)": insulin, "BMI (kg/m²)": bmi,
            "Diabetes Pedigree Function": dpf, "Age (years)": age
        }

        st.session_state.history.append({
            "time": datetime.now().strftime("%d %b %Y %H:%M"),
            "disease": "Diabetes", "score": score,
            "level": level,
            "prediction": "Diabetic" if pred==1 else "Non-Diabetic"
        })

        st.markdown("---")
        st.markdown("#### 📊 Prediction Results")
        r1,r2,r3 = st.columns(3)
        r1.metric("Risk Score",  f"{score}/100")
        r2.metric("Risk Level",  level)
        r3.metric("Prediction",  "Diabetic" if pred==1 else "Non-Diabetic")

        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(make_gauge(score, color, "Diabetes Risk Score"),
                            use_container_width=True)
        with g2:
            st.plotly_chart(make_risk_bar(score, "Diabetes"),
                            use_container_width=True)
            st.plotly_chart(make_feature_bar(
                list(inputs_dict.keys()),
                list(inputs_dict.values()),
                "Patient Input Parameters", color),
                use_container_width=True)

        result_label = "Diabetic" if pred==1 else "Non-Diabetic"
        st.markdown(f"""
        <div class="result-{css}">
          <h3>{icon} {level} — {result_label}</h3>
          <p><b>Risk Score:</b> {score}/100 &nbsp;|&nbsp;
             <b>Probability:</b> {round(prob*100,1)}%</p>
          <p><b>Clinical Recommendations:</b></p>
          <ul>{"".join(f"<li>{r}</li>" for r in recs)}</ul>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        report = generate_report("Diabetes", score, level,
                                  result_label, inputs_dict, recs)
        fname  = f"Diabetes_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.markdown(download_btn(report, fname), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — HEART DISEASE
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-title">❤️ 10-Year Heart Disease Risk Prediction</div>',
                unsafe_allow_html=True)
    st.caption("Enter the patient's clinical and lifestyle parameters below.")

    with st.form("heart_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            h_age    = st.number_input("🎂 Age (years)",          20, 90, 45, key="ha")
            h_male   = st.selectbox("👤 Gender",
                                    ["Female (0)","Male (1)"], key="hg")
            h_male   = 1 if "Male" in h_male else 0
            h_smoker = st.selectbox("🚬 Current Smoker",
                                    ["No (0)","Yes (1)"], key="hsk")
            h_smoker = 1 if "Yes" in h_smoker else 0
            h_cigs   = st.number_input("🚬 Cigarettes Per Day",   0, 60,  0, key="hc")
            h_bp_med = st.selectbox("💊 On BP Medication",
                                    ["No (0)","Yes (1)"], key="hb")
            h_bp_med = 1 if "Yes" in h_bp_med else 0
        with col2:
            h_stroke = st.selectbox("🧠 Prevalent Stroke",
                                    ["No (0)","Yes (1)"], key="hst")
            h_stroke = 1 if "Yes" in h_stroke else 0
            h_hyp    = st.selectbox("🩺 Hypertension",
                                    ["No (0)","Yes (1)"], key="hhy")
            h_hyp    = 1 if "Yes" in h_hyp else 0
            h_diab   = st.selectbox("🩸 Diabetes",
                                    ["No (0)","Yes (1)"], key="hd")
            h_diab   = 1 if "Yes" in h_diab else 0
            h_chol   = st.number_input("🧪 Total Cholesterol (mg/dL)",
                                       100, 600, 200, key="hch")
            h_edu    = st.selectbox("🎓 Education Level",
                                    ["1 - Some High School","2 - High School/GED",
                                     "3 - Some College","4 - College Degree"], key="he")
            h_edu    = int(h_edu[0])
        with col3:
            h_sysbp = st.number_input("💓 Systolic BP (mm Hg)",   80,  300, 120, key="hsy")
            h_diabp = st.number_input("💓 Diastolic BP (mm Hg)",  40,  150,  80, key="hdi")
            h_bmi   = st.number_input("⚖️  BMI (kg/m²)",          10.0,60.0,25.0, key="hbm")
            h_hr    = st.number_input("❤️  Heart Rate (bpm)",     40,  200,  75, key="hhr")
            h_gluc  = st.number_input("🩸 Glucose Level (mg/dL)", 40,  400,  80, key="hgl")
        h_submitted = st.form_submit_button("🔍 Predict Heart Disease Risk",
                                             use_container_width=True)

    if h_submitted:
        pulse_pressure = h_sysbp - h_diabp
        heavy_smoker   = 1 if h_cigs > 20 else 0
        age_risk       = 0 if h_age<40 else (1 if h_age<50 else (2 if h_age<60 else 3))
        bmi_category   = 0 if h_bmi<18.5 else (1 if h_bmi<25 else (2 if h_bmi<30 else 3))

        inp = np.array([[h_male, h_age, h_edu, h_smoker, h_cigs,
                         h_bp_med, h_stroke, h_hyp, h_diab, h_chol,
                         h_sysbp, h_diabp, h_bmi, h_hr, h_gluc,
                         pulse_pressure, heavy_smoker, age_risk, bmi_category]])
        scaled = heart_scaler.transform(inp)
        pred   = heart_model.predict(scaled)[0]
        prob   = heart_model.predict_proba(scaled)[0][1]
        score, level, css, icon, color = get_risk(prob)

        recs = (["Consult a cardiologist immediately",
                 "Monitor BP and cholesterol regularly",
                 "Quit smoking completely",
                 "Follow heart-healthy diet (low salt, low fat)",
                 "Exercise under medical supervision",
                 "Take prescribed medications consistently",
                 "Consider cardiac stress test and ECG"]
                if pred == 1 else
                ["Maintain current healthy lifestyle",
                 "Annual cardiac check-up recommended",
                 "Keep cholesterol and BP in normal range",
                 "Stay physically active (30 min/day)",
                 "Avoid smoking and excessive alcohol"])

        inputs_dict = {
            "Age (years)": h_age,
            "Gender": "Male" if h_male else "Female",
            "Smoker": "Yes" if h_smoker else "No",
            "Cigarettes/Day": h_cigs,
            "BP Medication": "Yes" if h_bp_med else "No",
            "Prior Stroke": "Yes" if h_stroke else "No",
            "Hypertension": "Yes" if h_hyp else "No",
            "Diabetes": "Yes" if h_diab else "No",
            "Total Cholesterol (mg/dL)": h_chol,
            "Systolic BP (mm Hg)": h_sysbp,
            "Diastolic BP (mm Hg)": h_diabp,
            "BMI (kg/m²)": h_bmi,
            "Heart Rate (bpm)": h_hr,
            "Glucose (mg/dL)": h_gluc,
            "Pulse Pressure (eng.)": pulse_pressure,
            "Age Risk Group (eng.)": age_risk,
            "BMI Category (eng.)": bmi_category
        }

        st.session_state.history.append({
            "time": datetime.now().strftime("%d %b %Y %H:%M"),
            "disease": "Heart Disease", "score": score,
            "level": level,
            "prediction": "CHD Risk" if pred==1 else "No CHD Risk"
        })

        st.markdown("---")
        st.markdown("#### 📊 Prediction Results")
        r1,r2,r3 = st.columns(3)
        r1.metric("10-Year CHD Risk Score", f"{score}/100")
        r2.metric("Risk Level",  level)
        r3.metric("Prediction",  "CHD Risk" if pred==1 else "No CHD Risk")

        g1, g2 = st.columns([1, 2])
        with g1:
            st.plotly_chart(make_gauge(score, color, "Heart Disease Risk Score"),
                            use_container_width=True)
        with g2:
            st.plotly_chart(make_risk_bar(score, "Heart Disease"),
                            use_container_width=True)
            num_inputs = {k: v for k,v in inputs_dict.items()
                          if isinstance(v, (int, float))}
            st.plotly_chart(make_feature_bar(
                list(num_inputs.keys()),
                list(num_inputs.values()),
                "Numerical Input Parameters", color),
                use_container_width=True)

        result_label = "CHD Risk" if pred==1 else "No CHD Risk"
        st.markdown(f"""
        <div class="result-{css}">
          <h3>{icon} {level} — {result_label}</h3>
          <p><b>10-Year CHD Risk Score:</b> {score}/100 &nbsp;|&nbsp;
             <b>Probability:</b> {round(prob*100,1)}%</p>
          <p><b>Clinical Recommendations:</b></p>
          <ul>{"".join(f"<li>{r}</li>" for r in recs)}</ul>
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        report = generate_report("Heart Disease (10-Year CHD)", score, level,
                                  result_label, inputs_dict, recs)
        fname  = f"HeartDisease_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        st.markdown(download_btn(report, fname), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — HISTORY
# ══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-title">🗂️ Prediction History</div>',
                unsafe_allow_html=True)

    if not st.session_state.history:
        st.info("No predictions yet. Make a prediction to see history here.")
    else:
        col_clr, col_exp, _ = st.columns([1,1,3])
        with col_clr:
            if st.button("🗑️ Clear History"):
                st.session_state.history = []
                st.rerun()
        with col_exp:
            df_hist = pd.DataFrame(st.session_state.history)
            csv = df_hist.to_csv(index=False).encode()
            st.download_button("📥 Export CSV", csv,
                               "prediction_history.csv", "text/csv")

        st.markdown(f"**Total Predictions: {len(st.session_state.history)}**")
        st.markdown("---")

        for i, h in enumerate(reversed(st.session_state.history)):
            badge = ("badge-high" if "HIGH" in h['level']
                     else "badge-moderate" if "MODERATE" in h['level']
                     else "badge-low")
            st.markdown(f"""
            <div class="history-card">
              <b>#{len(st.session_state.history)-i} &nbsp; {h['disease']}</b>
              &nbsp;&nbsp;<span class="{badge}">{h['level']}</span><br>
              <small style="color:#607D8B">
                🕐 {h['time']} &nbsp;|&nbsp;
                🎯 <b>{h['prediction']}</b> &nbsp;|&nbsp;
                📊 Score: <b>{h['score']}/100</b>
              </small>
            </div>""", unsafe_allow_html=True)

        if len(st.session_state.history) > 1:
            st.markdown("---")
            st.plotly_chart(make_trend(st.session_state.history),
                            use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 4 — ABOUT
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown('<div class="section-title">ℹ️ About This Project</div>',
                unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**🎯 Project Overview**

**AI-Powered Dual Disease Prediction System** developed for
B.Tech Project-I (CS3201) at IIIT Senapati, Manipur.

Predicts risk for two chronic diseases:
- 🩸 **Diabetes** — PIMA Indians Dataset (768 records)
- ❤️ **Heart Disease** — Framingham Study (4,240 records)

**🔬 Unique Contributions**
- Unified multi-disease prediction
- 0–100 Risk Score with severity levels
- Personalised clinical recommendations
- Feature engineering (pulse pressure, age risk etc.)
- SMOTE for class imbalance correction
- 7 ML models compared with ROC-AUC
- PDF report download & history tracking
        """)
    with c2:
        st.markdown("**📊 Model Performance**")
        st.dataframe(pd.DataFrame({
            "Model": ["Logistic Regression","Random Forest","SVM","KNN",
                      "XGBoost ⭐","Gradient Boosting","Voting Ensemble"],
            "Diabetes": ["71.43%","74.68%","72.73%","68.83%",
                         "76.62%","74.68%","75.97%"],
            "Heart Disease": ["66.98%","73.58%","68.75%","61.32%",
                              "81.37%","80.31%","75.24%"]
        }), use_container_width=True, hide_index=True)

        st.markdown("""
**👨‍💻 Developer**

| Field | Details |
|-------|---------|
| Name | Pritesh Raj |
| Roll No | 230103035 |
| Batch | 2023–2027 |
| Dept | CSE (AI & Data Science) |
| College | IIIT Senapati, Manipur |
| Supervisor | Dr. Nongmeikapam Kishorjit Singh |
        """)

    st.markdown("---")
    st.warning("⚠️ **Disclaimer:** For educational purposes only. "
               "Always consult a qualified medical professional for clinical diagnosis.")