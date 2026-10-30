"""Intelligent Financial Decision Support System - Streamlit frontend.
Built from model_training.ipynb. No model is retrained here: the saved models and
preprocessor are loaded, and the identical 80/20 stratified split (random_state=42)
is re-created from dataset.xls to evaluate them."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import shap
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Financial Decision Support", page_icon="◆", layout="wide")

# ----------------------------------------------------------------- notebook constants
BASE = Path(__file__).parent
TARGET = "default payment next month"
BILL_COLS = [f"BILL_AMT{i}" for i in range(1, 7)]
PAY_AMT_COLS = [f"PAY_AMT{i}" for i in range(1, 7)]
PAY_COLS = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
ENGINEERED = ["Total_Bill_Amount", "Total_Payment_Amount", "Average_Bill_Amount",
              "Average_Payment_Amount", "Payment_Ratio"]
CAT = ["SEX", "EDUCATION", "MARRIAGE"]
MODEL_FILES = {"lr": "logistic_regression_model.pkl", "rf": "random_forest_model.pkl",
               "xgb": "xgboost_model.pkl", "pre": "preprocessor.pkl", "names": "feature_names.pkl"}
METRICS = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]
LR, RF, XGB = "Logistic Regression", "Random Forest", "XGBoost"
# Risk thresholds exactly as in the notebook
HIGH, MEDIUM = 0.70, 0.40

LABELS = {
    "SEX": {1: "Male", 2: "Female"},
    "EDUCATION": {1: "Graduate School", 2: "University", 3: "High School", 4: "Others"},
    "MARRIAGE": {1: "Married", 2: "Single", 3: "Others"},
}
PAY_LABEL = {-2: "No consumption", -1: "Paid in full", 0: "Revolving credit"}

C_BG, C_CARD, C_BORDER = "#0B1020", "#121A33", "#222C4D"
C_LR, C_RF, C_XGB, C_UP, C_DOWN = "#64748B", "#10B981", "#6366F1", "#F87171", "#38BDF8"
RISK_COLOR = {"Low Risk": "#34D399", "Medium Risk": "#FBBF24", "High Risk": "#F87171"}


class AppError(Exception):
    pass


# ----------------------------------------------------------------- styling
st.markdown(f"""<style>
.block-container {{padding-top:2rem; max-width:1250px;}}
[data-testid="stSidebar"] {{background:{C_CARD}; border-right:1px solid {C_BORDER};}}
.hero {{padding:1.6rem 2rem; border-radius:16px; border:1px solid {C_BORDER};
  background:linear-gradient(135deg,#141C3A 0%,#1B2250 55%,#232A6B 100%); margin-bottom:1.2rem;}}
.hero h1 {{font-size:1.7rem; letter-spacing:.04em; margin:0; font-weight:700;}}
.hero p {{margin:.3rem 0 0; color:#A5B0D6; letter-spacing:.18em; font-size:.85rem; text-transform:uppercase;}}
.card {{background:{C_CARD}; border:1px solid {C_BORDER}; border-radius:14px; padding:1.1rem 1.3rem; height:100%;}}
.card .lbl {{color:#8F9BC4; font-size:.74rem; letter-spacing:.12em; text-transform:uppercase;}}
.card .val {{font-size:1.9rem; font-weight:700; margin-top:.15rem; line-height:1.2;}}
.card .sub {{color:#8F9BC4; font-size:.8rem; margin-top:.2rem;}}
.sec {{font-size:.8rem; letter-spacing:.14em; color:#8F9BC4; text-transform:uppercase;
  margin:1.6rem 0 .6rem; padding-bottom:.4rem; border-bottom:1px solid {C_BORDER};}}
.note {{background:#10173A; border-left:3px solid {C_XGB}; padding:.8rem 1rem; border-radius:6px;
  color:#C3CCEA; font-size:.9rem;}}
.step {{display:flex; gap:.8rem; align-items:flex-start; padding:.55rem 0; border-bottom:1px solid {C_BORDER};}}
.step b {{background:{C_XGB}; color:white; border-radius:50%; min-width:1.6rem; height:1.6rem;
  display:flex; align-items:center; justify-content:center; font-size:.8rem;}}
.stButton>button, .stFormSubmitButton>button {{background:{C_XGB}; color:white; border:0; border-radius:10px;
  padding:.65rem 1.6rem; font-weight:600; letter-spacing:.03em;}}
.stButton>button:hover, .stFormSubmitButton>button:hover {{background:#4F52D8; color:white;}}
footer, #MainMenu {{visibility:hidden;}}
</style>""", unsafe_allow_html=True)


def card(label, value, sub="", color="#E6EAF5"):
    return (f'<div class="card"><div class="lbl">{label}</div>'
            f'<div class="val" style="color:{color}">{value}</div><div class="sub">{sub}</div></div>')


def section(t):
    st.markdown(f'<div class="sec">{t}</div>', unsafe_allow_html=True)


def style(fig, h=380):
    fig.update_layout(height=h, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color="#C3CCEA"), margin=dict(l=10, r=10, t=40, b=10),
                      legend=dict(orientation="h", y=1.12, x=0))
    fig.update_xaxes(gridcolor=C_BORDER, zeroline=False)
    fig.update_yaxes(gridcolor=C_BORDER, zeroline=False)
    return fig


# ----------------------------------------------------------------- pipeline helpers
def _find(name):
    for p in (BASE / name, BASE / "models" / name):
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    missing = [f for f in MODEL_FILES.values() if _find(f) is None]
    if missing:
        raise AppError("Missing model file(s): " + ", ".join(missing) +
                       ". Run the final save cells of model_training.ipynb and place the .pkl "
                       "files next to app.py (or in a 'models' folder).")
    m = {k: joblib.load(_find(f)) for k, f in MODEL_FILES.items()}
    m["explainer"] = shap.TreeExplainer(m["xgb"])  # same as notebook
    return m


def add_engineered_features(df):
    """Identical to the notebook's feature-engineering cells."""
    df = df.copy()
    df["Total_Bill_Amount"] = df[BILL_COLS].sum(axis=1)
    df["Total_Payment_Amount"] = df[PAY_AMT_COLS].sum(axis=1)
    df["Average_Bill_Amount"] = df[BILL_COLS].mean(axis=1)
    df["Average_Payment_Amount"] = df[PAY_AMT_COLS].mean(axis=1)
    df["Payment_Ratio"] = df["Total_Payment_Amount"] / (df["Total_Bill_Amount"] + 1)
    return df


@st.cache_data(show_spinner="Loading dataset...")
def load_dataset():
    path = next((p for n in ("dataset.xls", "dataset.xlsx") for p in (BASE / n, BASE / "data" / n)
                 if p.exists()), None)
    if path is None:
        raise AppError("Dataset file 'dataset.xls' not found. Place the same file used in the "
                       "notebook next to app.py (or in a 'data' folder).")
    df = pd.read_excel(path, header=1)                       # notebook: header=1
    df = df.drop_duplicates().reset_index(drop=True)
    df = df.drop(columns=["ID"])
    return add_engineered_features(df)


@st.cache_data(show_spinner="Evaluating models on the held-out test set...")
def evaluate_all():
    """Re-creates the notebook's test split (test_size=0.20, random_state=42, stratify=y)
    and evaluates the SAVED models. SMOTE is applied only to training data, so it is not needed."""
    m, df = load_models(), load_dataset()
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    _, X_te, _, y_te = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    Xp = m["pre"].transform(X_te)
    out = {"y_true": y_te.values, "n_test": len(y_te), "Xp": Xp}
    for key, name in (("lr", LR), ("rf", RF), ("xgb", XGB)):
        pred, prob = m[key].predict(Xp), m[key].predict_proba(Xp)[:, 1]
        out[name] = {"pred": pred, "prob": prob, "metrics": calculate_metrics(y_te, pred, prob),
                     "cm": confusion_matrix(y_te, pred), "roc": roc_curve(y_te, prob)}
    return out


def calculate_metrics(y_true, pred, prob):
    return {"Accuracy": accuracy_score(y_true, pred), "Precision": precision_score(y_true, pred),
            "Recall": recall_score(y_true, pred), "F1 Score": f1_score(y_true, pred),
            "ROC-AUC": roc_auc_score(y_true, prob)}


def pretty(name):
    """'num__PAY_0' -> 'PAY_0';  'cat__SEX_2' -> 'SEX = Female'."""
    n = name.split("__", 1)[-1]
    for c in CAT:
        if n.startswith(c + "_"):
            try:
                v = int(float(n[len(c) + 1:]))
                return f"{c} = {LABELS[c].get(v, 'Undocumented ' + str(v))}"
            except ValueError:
                pass
    return n


def risk_level(p):
    return "High Risk" if p >= HIGH else "Medium Risk" if p >= MEDIUM else "Low Risk"


def prepare_input(raw):
    """Raw form values -> engineered DataFrame in the exact column order seen at training."""
    pre = load_models()["pre"]
    df = add_engineered_features(pd.DataFrame([raw]))
    expected = list(getattr(pre, "feature_names_in_", []))
    if expected:
        missing = [c for c in expected if c not in df.columns]
        if missing:
            raise AppError("Feature mismatch: the saved preprocessor expects columns that the form "
                           "does not provide: " + ", ".join(missing))
        df = df[expected]
    return df


def generate_prediction(raw):
    """Same logic as notebook predict_credit_risk() + explain_customer()."""
    m = load_models()
    Xp = m["pre"].transform(prepare_input(raw))
    prob = float(m["xgb"].predict_proba(Xp)[0, 1])
    pred = int(m["xgb"].predict(Xp)[0])
    sv = np.asarray(m["explainer"].shap_values(Xp))[0]
    exp = pd.DataFrame({"Feature": [pretty(n) for n in m["names"]], "SHAP_Value": sv,
                        "Feature_Value": Xp[0]})
    exp["Absolute_SHAP"] = exp["SHAP_Value"].abs()
    exp = exp.sort_values("Absolute_SHAP", ascending=False).reset_index(drop=True)
    return {"raw": raw, "prob": prob, "pred": pred, "risk": risk_level(prob), "explanation": exp}


@st.cache_data(show_spinner="Computing global SHAP values...")
def global_shap():
    m, ev = load_models(), evaluate_all()
    sv = np.asarray(m["explainer"].shap_values(ev["Xp"]))
    return (pd.DataFrame({"Feature": [pretty(n) for n in m["names"]], "Mean_SHAP": np.abs(sv).mean(axis=0)})
            .sort_values("Mean_SHAP", ascending=False).reset_index(drop=True))


def generate_shap_explanation(res, top=3):
    e = res["explanation"]
    return (e[e.SHAP_Value > 0].head(top)["Feature"].tolist(),
            e[e.SHAP_Value < 0].head(top)["Feature"].tolist())


DECISION_TEXT = {  # verbatim from notebook get_final_decision_support()
    "High Risk": "The customer has a high predicted default risk. Review repayment behaviour, outstanding "
                 "balances, and recent payment patterns before making a credit decision.",
    "Medium Risk": "The customer has a moderate predicted default risk. Regular monitoring of repayment "
                   "behaviour and payment patterns is recommended.",
    "Low Risk": "The customer has a relatively low predicted default risk. Current financial and repayment "
                "patterns appear comparatively stable.",
}


# ----------------------------------------------------------------- charts
def create_confusion_matrix(cm, title):
    (tn, fp), (fn, tp) = cm
    txt = [[f"True Negative<br><b>{tn:,}</b>", f"False Positive<br><b>{fp:,}</b>"],
           [f"False Negative<br><b>{fn:,}</b>", f"True Positive<br><b>{tp:,}</b>"]]
    fig = go.Figure(go.Heatmap(z=cm, x=["No Default", "Default"], y=["No Default", "Default"],
                               text=txt, texttemplate="%{text}", textfont=dict(size=14, color="white"),
                               colorscale=[[0, "#16204A"], [1, "#4F52D8"]], showscale=False))
    fig.update_layout(title=title, xaxis_title="Predicted", yaxis_title="Actual",
                      yaxis=dict(autorange="reversed"))
    return style(fig, 360)


def create_roc_curve(ev):
    fig = go.Figure()
    for name, c in ((LR, C_LR), (RF, C_RF), (XGB, C_XGB)):
        fpr, tpr, _ = ev[name]["roc"]
        fig.add_scatter(x=fpr, y=tpr, mode="lines", line=dict(color=c, width=3),
                        name=f"{name} — AUC {ev[name]['metrics']['ROC-AUC']:.3f}")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(color="#475569", dash="dash"),
                    name="Random guess")
    fig.update_layout(xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
    return style(fig, 430)


def bar_h(df, x, y, color, title):
    d = df.iloc[::-1]
    fig = go.Figure(go.Bar(x=d[x], y=d[y], orientation="h", marker_color=color))
    fig.update_layout(title=title)
    return style(fig, 420)


# ----------------------------------------------------------------- pages
def page_dashboard():
    df, ev = load_dataset(), evaluate_all()
    st.markdown('<div class="hero"><h1>INTELLIGENT FINANCIAL DECISION SUPPORT SYSTEM</h1>'
                '<p>Predict &nbsp;•&nbsp; Explain &nbsp;•&nbsp; Support</p></div>', unsafe_allow_html=True)
    best = max((LR, RF, XGB), key=lambda n: ev[n]["metrics"]["ROC-AUC"])
    n_feat = df.shape[1] - 1
    cols = st.columns(4)
    for c, h in zip(cols, [card("Customer records", f"{len(df):,}", "after de-duplication"),
                           card("Input features", n_feat, f"{n_feat - len(ENGINEERED)} original + {len(ENGINEERED)} engineered"),
                           card("ML models", 3, f"{LR} • {RF} • {XGB}"),
                           card("Best model (ROC-AUC)", best, f"AUC {ev[best]['metrics']['ROC-AUC']:.3f}", C_XGB)]):
        c.markdown(h, unsafe_allow_html=True)
    st.markdown('<div class="note" style="margin-top:1rem">Our system predicts credit default risk using machine '
                'learning, explains the prediction using SHAP, and converts the result into understandable '
                'decision-support guidance.</div>', unsafe_allow_html=True)

    section("Target distribution & dataset statistics")
    a, b = st.columns([1, 1.2])
    vc = df[TARGET].value_counts().sort_index()
    fig = go.Figure(go.Pie(labels=["No Default", "Default"], values=[vc.get(0, 0), vc.get(1, 0)], hole=.62,
                           marker_colors=["#4F52D8", "#F87171"], textinfo="percent"))
    a.plotly_chart(style(fig, 340), width="stretch")
    stats = pd.DataFrame({
        "Statistic": ["Default rate", "Non-default customers", "Default customers", "Mean credit limit",
                      "Median credit limit", "Mean age", "Mean total bill amount", "Mean total payment amount"],
        "Value": [f"{df[TARGET].mean():.2%}", f"{vc.get(0, 0):,}", f"{vc.get(1, 0):,}", f"{df.LIMIT_BAL.mean():,.0f}",
                  f"{df.LIMIT_BAL.median():,.0f}", f"{df.AGE.mean():.1f}", f"{df.Total_Bill_Amount.mean():,.0f}",
                  f"{df.Total_Payment_Amount.mean():,.0f}"]})
    b.dataframe(stats, hide_index=True, width="stretch")

    section("Machine-learning workflow (as implemented in the notebook)")
    steps = ["Load dataset and remove duplicates and the ID column",
             "Engineer 5 features: total / average bill, total / average payment, payment ratio",
             "Stratified 80/20 train-test split (random_state = 42)",
             "StandardScaler on numerical features, One-Hot Encoding on SEX, EDUCATION, MARRIAGE",
             "SMOTE applied to the training set only to balance the classes",
             f"Train {LR}, {RF}, and {XGB}",
             "Evaluate: Accuracy, Precision, Recall, F1, ROC-AUC, confusion matrix",
             "SHAP TreeExplainer on XGBoost, then rule-based decision-support guidance"]
    st.markdown("".join(f'<div class="step"><b>{i}</b><span>{s}</span></div>' for i, s in enumerate(steps, 1)),
                unsafe_allow_html=True)


def page_evaluation():
    ev = evaluate_all()
    st.title("Model Evaluation")
    st.caption(f"Computed live from the saved models on the notebook's held-out test set ({ev['n_test']:,} customers).")
    choice = st.selectbox("SELECT EVALUATION METRIC", ["All"] + METRICS)
    models = (LR, RF, XGB)
    M = {n: ev[n]["metrics"] for n in models}
    if choice == "All":
        table_data = {"Metric": METRICS}
        for n in models:
            table_data[n] = [f"{M[n][k]:.2%}" if k != "ROC-AUC" else f"{M[n][k]:.4f}" for k in METRICS]
        table = pd.DataFrame(table_data)
        st.dataframe(table, hide_index=True, width="stretch")
        fig = go.Figure()
        for n, c in ((LR, C_LR), (RF, C_RF), (XGB, C_XGB)):
            fig.add_bar(name=n, x=METRICS, y=[M[n][k] for k in METRICS], marker_color=c,
                        text=[f"{M[n][k]:.3f}" for k in METRICS], textposition="outside")
        fig.update_layout(barmode="group", yaxis=dict(range=[0, 1.1], title="Score"))
        st.plotly_chart(style(fig, 420), width="stretch")
    else:
        fmt = (lambda v: f"{v:.4f}") if choice == "ROC-AUC" else (lambda v: f"{v:.2%}")
        win = max(models, key=lambda n: M[n][choice])
        cols = st.columns(3)
        for c, n in zip(cols, models):
            c.markdown(card(n, fmt(M[n][choice]), f"{choice}" + ("  •  higher" if n == win else ""),
                            C_XGB if n == win else "#E6EAF5"), unsafe_allow_html=True)
        fig = go.Figure(go.Bar(x=list(models), y=[M[n][choice] for n in models],
                               marker_color=[C_LR, C_RF, C_XGB],
                               text=[fmt(M[n][choice]) for n in models], textposition="outside"))
        fig.update_layout(yaxis=dict(range=[0, 1.1], title=choice), title=f"{choice} comparison")
        st.plotly_chart(style(fig, 380), width="stretch")

    section("Confusion matrices")
    c1, c2, c3 = st.columns(3)
    c1.plotly_chart(create_confusion_matrix(ev[LR]["cm"], f"{LR}"), width="stretch")
    c2.plotly_chart(create_confusion_matrix(ev[RF]["cm"], f"{RF}"), width="stretch")
    c3.plotly_chart(create_confusion_matrix(ev[XGB]["cm"], f"{XGB}"), width="stretch")

    section("ROC curve")
    st.plotly_chart(create_roc_curve(ev), width="stretch")
    st.markdown('<div class="note">ROC-AUC measures how well a model separates defaulters from non-defaulters '
                'across all thresholds. A higher AUC indicates better class discrimination; 0.5 is random guessing.</div>',
                unsafe_allow_html=True)


def page_prediction():
    df = load_dataset()
    st.title("Credit Risk Prediction")
    st.caption("Engineered features are calculated automatically from the values you enter.")

    def cat_select(col, label):
        opts = sorted(int(v) for v in df[col].unique())
        return st.selectbox(label, opts, format_func=lambda v: LABELS[col].get(v, f"Undocumented ({v})"))

    def pay_select(col):
        opts = sorted(int(v) for v in df[col].unique())
        default = opts.index(int(df[col].mode()[0]))
        return st.selectbox(col, opts, index=default,
                            format_func=lambda v: f"{v} — {PAY_LABEL.get(v, f'{v} month delay')}")

    raw = {}
    with st.form("customer"):
        section("Customer profile")
        c = st.columns(5)
        raw["LIMIT_BAL"] = c[0].number_input("Credit Limit", 0, value=int(df.LIMIT_BAL.median()), step=10000)
        with c[1]: raw["SEX"] = cat_select("SEX", "Gender")
        with c[2]: raw["EDUCATION"] = cat_select("EDUCATION", "Education")
        with c[3]: raw["MARRIAGE"] = cat_select("MARRIAGE", "Marriage")
        raw["AGE"] = c[4].number_input("Age", 18, 100, int(df.AGE.median()))

        section("Repayment behaviour (-2 no consumption, -1 paid in full, 0 revolving, 1+ months delayed)")
        c = st.columns(6)
        for col, w in zip(PAY_COLS, c):
            with w: raw[col] = pay_select(col)

        section("Bill amounts")
        c = st.columns(6)
        for col, w in zip(BILL_COLS, c):
            raw[col] = w.number_input(col, value=int(df[col].median()), step=1000)

        section("Payment amounts")
        c = st.columns(6)
        for col, w in zip(PAY_AMT_COLS, c):
            raw[col] = w.number_input(col, 0, value=int(df[col].median()), step=500)
        go_btn = st.form_submit_button("Run Credit Risk Assessment")

    if go_btn:
        st.session_state["result"] = generate_prediction(raw)
    res = st.session_state.get("result")
    if not res:
        return
    color = RISK_COLOR[res["risk"]]
    section("Assessment result")
    a, b, c3 = st.columns(3)
    a.markdown(card("Risk level", res["risk"].upper(), "thresholds: ≥70% high, ≥40% medium", color), unsafe_allow_html=True)
    b.markdown(card("Default probability", f"{res['prob']:.2%}", "predicted by XGBoost"), unsafe_allow_html=True)
    c3.markdown(card("Model used", XGB, "primary prediction model", C_XGB), unsafe_allow_html=True)
    fig = go.Figure(go.Indicator(mode="gauge+number", value=res["prob"] * 100,
                                 number=dict(suffix="%", valueformat=".2f"),
                                 gauge=dict(axis=dict(range=[0, 100]), bar=dict(color=color),
                                            bgcolor=C_CARD, borderwidth=0,
                                            steps=[dict(range=[0, 40], color="#14283A"),
                                                   dict(range=[40, 70], color="#2B2A2E"),
                                                   dict(range=[70, 100], color="#35202B")])))
    st.plotly_chart(style(fig, 280), width="stretch")
    st.caption("Open SHAP Explainability and Decision Support from the sidebar for the explanation.")


def _need_result():
    res = st.session_state.get("result")
    if not res:
        st.info("No customer assessed yet. Go to **Credit Risk Prediction** and run an assessment first.")
    return res


def page_shap():
    st.title("SHAP Explainability")
    st.caption("Why did the model make this prediction? SHAP TreeExplainer on the XGBoost model.")
    section("Global SHAP explanation")
    g = global_shap()
    st.plotly_chart(bar_h(g.head(15), "Mean_SHAP", "Feature", C_XGB, "Top 15 features by mean |SHAP| (test set)"),
                    width="stretch")
    section("Individual customer explanation")
    res = _need_result()
    if not res:
        return
    e = res["explanation"]
    st.markdown(f"Predicted default probability: **{res['prob']:.2%}** — **{res['risk']}**")
    up, down = e[e.SHAP_Value > 0].head(7), e[e.SHAP_Value < 0].head(7)
    a, b = st.columns(2)
    a.plotly_chart(bar_h(up, "SHAP_Value", "Feature", C_UP, "Risk-increasing factors"), width="stretch")
    d = down.assign(SHAP_Value=down.SHAP_Value.abs())
    b.plotly_chart(bar_h(d, "SHAP_Value", "Feature", C_DOWN, "Risk-reducing factors (magnitude)"), width="stretch")
    st.markdown('<div class="note">Positive SHAP values push the prediction towards default; negative values push it '
                'away. Values are in the model\'s log-odds space.</div>', unsafe_allow_html=True)
    with st.expander("Full SHAP table for this customer"):
        st.dataframe(e[["Feature", "SHAP_Value", "Feature_Value"]], hide_index=True, width="stretch")


def page_decision():
    st.title("Decision Support")
    res = _need_result()
    if not res:
        return
    inc, red = generate_shap_explanation(res)
    color = RISK_COLOR[res["risk"]]
    a, b = st.columns(2)
    a.markdown(card("Risk assessment", res["risk"].upper(), "", color), unsafe_allow_html=True)
    b.markdown(card("Default probability", f"{res['prob']:.2%}", "XGBoost"), unsafe_allow_html=True)
    section("Why?")
    a, b = st.columns(2)
    a.markdown("**Important risk factors**\n\n" + ("\n".join(f"- {f}" for f in inc) or "- None identified"))
    b.markdown("**Important risk-reducing factors**\n\n" + ("\n".join(f"- {f}" for f in red) or "- None identified"))
    section("Decision support")
    st.markdown(f'<div class="note">{DECISION_TEXT[res["risk"]]}</div>', unsafe_allow_html=True)
    st.warning("This is a decision-support prototype built for academic purposes. It does not replace a financial "
               "institution's final credit decision.")


PAGES = {"Dashboard": page_dashboard, "Model Evaluation": page_evaluation,
         "Credit Risk Prediction": page_prediction, "SHAP Explainability": page_shap,
         "Decision Support": page_decision}

with st.sidebar:
    st.markdown("### ◆ FinDSS")
    st.caption("Predict • Explain • Support")
    page = st.radio("Navigation", list(PAGES), label_visibility="collapsed")
    st.divider()
    st.caption("B.Tech AI & ML Project\nCredit default risk • Logistic Regression, Random Forest & XGBoost")

try:
    PAGES[page]()
except AppError as err:
    st.error(str(err))
except Exception as err:  # no raw tracebacks for end users
    st.error(f"Something went wrong while running this page ({type(err).__name__}). "
             f"If this appeared after prediction, check that the saved model files match the dataset "
             f"columns and the package versions used in the notebook. Details: {err}")
