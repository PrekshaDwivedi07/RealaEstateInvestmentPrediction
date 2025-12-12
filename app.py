# app.py
import os
import textwrap
import joblib
import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# ---------------------------
# Helper: create suggestion images (must be defined before any use)
# ---------------------------
def create_img(title, lines, size=(1000, 220)):
    """
    Small Pillow-based image generator for footer suggestions.
    Returns a PIL Image object.
    """
    img = Image.new("RGB", size, (240, 244, 248))
    d = ImageDraw.Draw(img)
    try:
        tfont = ImageFont.truetype("DejaVuSans-Bold.ttf", 20)
        bfont = ImageFont.truetype("DejaVuSans.ttf", 14)
    except Exception:
        tfont = ImageFont.load_default()
        bfont = ImageFont.load_default()
    d.text((12, 8), title, fill=(10, 80, 80), font=tfont)
    y = 42
    for line in lines:
        for wrapped in textwrap.wrap(line, width=100):
            d.text((12, y), wrapped, fill=(25, 25, 25), font=bfont)
            y += 18
        y += 6
    return img

# ---------------------------
# Page config and theme CSS
# ---------------------------
st.set_page_config(page_title="Real Estate Agent Dashboard", layout="wide", initial_sidebar_state="expanded")

# Colors (tweak these for theme)
BG = "#18343a"          # deep teal/dark
PANEL = "#143033"       # slightly darker panel
CARD = "#122a2c"
TEXT = "#c1e8e6"        # light cyan for headings
SUB = "#8fd9d6"         # subtitle color
KPI = "#ffb74d"         # warm accent
POS = "#53b788"
NEG = "#ef6a6a"

# Inject CSS to mimic look
st.markdown(f"""
    <style>
    .reportview-container {{
        background: {BG};
    }}
    .stApp {{
        background: linear-gradient(180deg, {BG}, #0f2526);
    }}
    .block-container{{ padding-top:1rem; padding-left:1.4rem; padding-right:1.4rem; max-width:1700px;}}
    h1, h2, h3, .css-1v3fvcr h1 {{ color: {TEXT}; }}
    .stButton>button {{ background-color: #1f4e4d; color: {TEXT}; border-radius:6px; padding:8px 12px; }}
    .stMetric>div>div>p {{ color: {TEXT}; }}
    .stMetric>div>div>div {{ color: {KPI}; font-weight:700; }}
    .sidebar .sidebar-content {{
        background-color: {PANEL};
        color: {TEXT};
    }}
    .css-1d391kg {{ color: {SUB}; }}
    .stSelectbox>div>div>div {{ color: {TEXT}; background-color: #0d2a2a }}
    .element-container {{ padding: 0.25rem 0.5rem; }}
    </style>
""", unsafe_allow_html=True)

# ---------------------------
# Utility: Safe OneHotEncoder factory
# ---------------------------
def make_one_hot_encoder_safe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

# ---------------------------
# Load data (try couple of paths)
# ---------------------------
@st.cache_data
def load_data_try(paths):
    for p in paths:
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    raise FileNotFoundError(f"Dataset not found in: {paths}")

csv_paths = ["/mnt/data/india_housing_prices.csv", "india_housing_prices.csv", "./india_housing_prices.csv"]
try:
    df = load_data_try(csv_paths)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# Basic cleaning
for c in ["Price_in_Lakhs","Size_in_SqFt","BHK"]:
    if c in df.columns:
        df = df.dropna(subset=[c])
if "BHK" in df.columns:
    df["BHK"] = df["BHK"].astype(int)
if "Size_in_SqFt" in df.columns:
    df["Size_in_SqFt"] = pd.to_numeric(df["Size_in_SqFt"], errors="coerce")
if "Price_per_SqFt" not in df.columns and "Price_in_Lakhs" in df.columns and "Size_in_SqFt" in df.columns:
    df["Price_per_SqFt"] = (df["Price_in_Lakhs"] * 100000) / df["Size_in_SqFt"]
for cat in ["City","Property_Type","Furnished_Status","Locality"]:
    if cat in df.columns:
        df[cat] = df[cat].fillna("Unknown").astype(str)

# ---------------------------
# Sidebar (left nav look)
# ---------------------------
st.sidebar.markdown(f"# <span style='color:{TEXT}'>Real Estate Workspace</span>", unsafe_allow_html=True)
st.sidebar.markdown("---")
st.sidebar.header("Real Estate Agent Dashboard")
st.sidebar.markdown("Filters & model options")

city_opts = ["All"] + sorted(df["City"].unique().tolist())
city_filter = st.sidebar.selectbox("Date range (simulated)", ["Last 4 quarters","Last 12 months","Year to Date"])
city_select = st.sidebar.selectbox("City", city_opts, index=0)
prop_type_filter = st.sidebar.selectbox("Property Type", ["All"] + sorted(df["Property_Type"].unique().tolist()))
mode = st.sidebar.selectbox("Model speed/accuracy", ["fast","balanced","accurate"], index=0)
use_joblib = st.sidebar.checkbox("Use saved model (joblib)", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Built with ❤️ — Your Dashboard")

# ---------------------------
# Top layout: KPI columns under header area
# ---------------------------
st.title("Real Estate Agent Dashboard")

# Prep visuals dataframe per filters
df_vis = df.copy()
if city_select != "All":
    df_vis = df_vis[df_vis["City"] == city_select]
if prop_type_filter != "All":
    df_vis = df_vis[df_vis["Property_Type"] == prop_type_filter]

top_col1, top_col2, top_col3 = st.columns([1,1,1.1])

# KPI values (use available data or fallbacks)
avg_days_to_sale = int(df_vis.get("Age_of_Property", pd.Series([48])).median())
closed_deals = int(df_vis.shape[0] * 0.02)
avg_price = df_vis["Price_in_Lakhs"].mean()

with top_col1:
    st.markdown(f"<div style='color:{SUB};'>Average Time from Listing to Sale</div>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color:{KPI}; margin:0'>{avg_days_to_sale} days</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{NEG};'>-9.4% <span style='color:{SUB};'>Versus</span> </div>", unsafe_allow_html=True)
    st.markdown("<br>")
    st.markdown(f"<div style='color:{SUB};'>Number of Closed Deals</div>", unsafe_allow_html=True)
    st.markdown(f"<h1 style='color:{KPI}; margin:0'>{closed_deals}</h1>", unsafe_allow_html=True)
    st.markdown(f"<div style='color:{NEG};'>-4.1% <span style='color:{SUB};'>Versus</span> <span style='color:{POS};'>98 previous</span></div>", unsafe_allow_html=True)

# Stacked bar
with top_col2:
    st.markdown(f"<div style='color:{TEXT}; font-weight:600'>Number of Offers and Closed Deals</div>", unsafe_allow_html=True)
    q_labels = ["Q2 2024","Q3 2024","Q4 2024","Q1 2025"]
    base = [int(len(df_vis)*0.0002 + 20 + i) for i in range(4)]
    offers = [int(x*1.6) for x in base]
    closed = base
    fig_sb, ax_sb = plt.subplots(figsize=(5,3))
    ax_sb.bar(q_labels, closed, label="Closed", color="#1b9e77")
    ax_sb.bar(q_labels, offers, bottom=closed, label="Offers", color="#2fa7d6")
    ax_sb.set_ylabel("Number of Transactions", color=TEXT)
    ax_sb.legend(facecolor=CARD)
    ax_sb.set_facecolor(BG)
    fig_sb.patch.set_facecolor(BG)
    for spine in ax_sb.spines.values(): spine.set_visible(False)
    ax_sb.tick_params(colors=SUB)
    st.pyplot(fig_sb, use_container_width=True)

# Funnel
with top_col3:
    st.markdown(f"<div style='color:{TEXT}; font-weight:600'>Meeting-To-Deal Conversion</div>", unsafe_allow_html=True)
    stages = ["Client Meetings","Offers Made","Deal Closed"]
    vals = [267,185,94]
    fig_funnel = go.Figure(go.Funnel(y=stages, x=vals, marker={"color":["#2fa7d6","#53b788","#ef6a6a"]}))
    fig_funnel.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, font_color=TEXT, margin=dict(t=10,b=10,l=10,r=10))
    st.plotly_chart(fig_funnel, use_container_width=True)

# ---------------------------
# Next row: line chart, client satisfaction, commission heatmap
# ---------------------------
row1_col1, row1_col2, row1_col3 = st.columns([1,1,1])

with row1_col1:
    st.markdown(f"<div style='color:{TEXT}; font-weight:600'>Average Time From Listing To Sale</div>", unsafe_allow_html=True)
    months = pd.date_range(end=pd.Timestamp.today(), periods=12, freq='M')
    np.random.seed(42)
    vals = (np.linspace(500,495,12) + np.random.normal(0,6,12)).tolist()
    fig_ln, ax_ln = plt.subplots(figsize=(4.5,3))
    ax_ln.plot(months.strftime("%b %Y"), vals, marker='o', color="#2fa7d6")
    ax_ln.set_xticklabels(months.strftime("%b %Y"), rotation=45, fontsize=8)
    ax_ln.set_ylabel("Average Time To Sale", color=TEXT)
    ax_ln.set_facecolor(BG)
    fig_ln.patch.set_facecolor(BG)
    for spine in ax_ln.spines.values(): spine.set_visible(False)
    ax_ln.tick_params(colors=SUB)
    st.pyplot(fig_ln, use_container_width=True)

with row1_col2:
    st.markdown(f"<div style='color:{TEXT}; font-weight:600'>Client Satisfaction Scores</div>", unsafe_allow_html=True)
    agents = ["Agent 1","Agent 2","Agent 3","Agent 4","Agent 5","Agent 6"]
    scores = [3.0,2.9,4.2,4.0,3.3,2.8]
    fig_hb, ax_hb = plt.subplots(figsize=(4.5,3))
    ax_hb.barh(agents, scores, color=["#2fa7d6","#ef6a6a","#53b788","#53b788","#2fa7d6","#ef6a6a"])
    ax_hb.set_xlim(0,5)
    ax_hb.set_xlabel("Score")
    ax_hb.set_facecolor(BG)
    fig_hb.patch.set_facecolor(BG)
    for spine in ax_hb.spines.values(): spine.set_visible(False)
    ax_hb.tick_params(colors=SUB)
    st.pyplot(fig_hb, use_container_width=True)

with row1_col3:
    st.markdown(f"<div style='color:{TEXT}; font-weight:600'>Average Commission per Transaction</div>", unsafe_allow_html=True)
    prop_types = df["Property_Type"].unique()[:3].tolist() if len(df["Property_Type"].unique())>=3 else ["Luxury","Commercial","Residential"]
    agents_small = ["Agent 3","Agent 1","Agent 6","Agent 4","Agent 5","Agent 2"]
    np.random.seed(7)
    mat = np.abs(np.random.randn(len(agents_small), len(prop_types))) * 10 + 2
    fig_hm2, ax_hm2 = plt.subplots(figsize=(4.5,3))
    sns.heatmap(mat, annot=True, fmt=".1f", yticklabels=agents_small, xticklabels=prop_types, cmap="Blues", ax=ax_hm2, cbar=False)
    ax_hm2.set_facecolor(BG)
    fig_hm2.patch.set_facecolor(BG)
    for spine in ax_hm2.spines.values(): spine.set_visible(False)
    ax_hm2.tick_params(colors=SUB)
    st.pyplot(fig_hm2, use_container_width=True)

# ---------------------------
# Mid large heatmap
# ---------------------------
st.markdown("---")
st.markdown(f"<h3 style='color:{TEXT}'>Location heatmap (City × BHK — Avg Price per SqFt)</h3>", unsafe_allow_html=True)

top_cities = df_vis['City'].value_counts().head(12).index
pivot = df_vis[df_vis['City'].isin(top_cities)].pivot_table(index='City', columns='BHK', values='Price_per_SqFt', aggfunc='mean').fillna(0)
fig_p, ax_p = plt.subplots(figsize=(14, max(3, 0.4 * pivot.shape[0])))
sns.heatmap(pivot, annot=False, cmap='YlGnBu', ax=ax_p, cbar_kws={'shrink':0.7})
ax_p.set_ylabel("")
ax_p.set_facecolor(BG)
fig_p.patch.set_facecolor(BG)
for spine in ax_p.spines.values(): spine.set_visible(False)
ax_p.tick_params(colors=SUB)
st.pyplot(fig_p, use_container_width=True)

# ---------------------------
# Prediction panel
# ---------------------------
st.markdown("---")
st.markdown(f"<h3 style='color:{TEXT}'>Property investment simulator</h3>", unsafe_allow_html=True)
pcol1, pcol2 = st.columns([1,1])

with pcol1:
    city_in = st.selectbox("City (input)", sorted(df["City"].unique().tolist()))
    locality_in = st.text_input("Locality", value="Locality_1")
    prop_type_in = st.selectbox("Property Type", sorted(df["Property_Type"].unique().tolist()))
    bhk_in = st.selectbox("BHK", sorted(df["BHK"].unique().tolist()))
    size_in = st.number_input("Size (SqFt)", min_value=100, max_value=20000, value=1200)

with pcol2:
    year_built = st.number_input("Year Built", 1900, 2025, 2015)
    floor_no = st.number_input("Floor No", 0, 50, 1)
    total_floors = st.number_input("Total Floors", 1, 50, 10)
    furnished = st.selectbox("Furnished Status", sorted(df["Furnished_Status"].unique().tolist()))

predict_button = st.button("Predict & Evaluate")

# ---------------------------
# Model train & predict
# ---------------------------
MODEL_FILE = "rf_model_agent.joblib"

def train_model(train_df, mode="fast"):
    if mode == "fast":
        sample = min(5000, len(train_df))
        n_estimators = 50
        max_depth = 12
    elif mode == "balanced":
        sample = min(20000, len(train_df))
        n_estimators = 100
        max_depth = 16
    else:
        sample = len(train_df)
        n_estimators = 150
        max_depth = None

    if sample < len(train_df):
        tr = train_df.sample(n=sample, random_state=42)
    else:
        tr = train_df.copy()

    X = tr[["City","Property_Type","BHK","Size_in_SqFt","Year_Built","Furnished_Status","Floor_No","Total_Floors"]].copy()
    X["BHK"] = X["BHK"].astype(int)
    X["Size_in_SqFt"] = pd.to_numeric(X["Size_in_SqFt"], errors="coerce").fillna(X["Size_in_SqFt"].median())
    X["Year_Built"] = pd.to_numeric(X["Year_Built"], errors="coerce").fillna(X["Year_Built"].median())
    X["Floor_No"] = pd.to_numeric(X["Floor_No"], errors="coerce").fillna(0)
    X["Total_Floors"] = pd.to_numeric(X["Total_Floors"], errors="coerce").fillna(1)
    y = tr["Price_in_Lakhs"].astype(float).values

    encoder = make_one_hot_encoder_safe()
    pre = ColumnTransformer([("cat", encoder, ["City","Property_Type","Furnished_Status"])], remainder="passthrough")
    rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=42, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("rf", rf)])
    with st.spinner(f"Training model ({mode})..."):
        pipe.fit(X, y)

    try:
        ohe = pipe.named_steps["pre"].named_transformers_["cat"]
        cat_feats = list(ohe.get_feature_names_out(["City","Property_Type","Furnished_Status"]))
    except Exception:
        cat_feats = []
    num_cols = ["BHK","Size_in_SqFt","Year_Built","Floor_No","Total_Floors"]
    features = cat_feats + num_cols
    importances = pipe.named_steps["rf"].feature_importances_
    fi = pd.Series(importances, index=features[:len(importances)]).sort_values(ascending=False)
    return pipe, fi

# load or train
model_pipe = None
feature_imp = None
if use_joblib and os.path.exists(MODEL_FILE):
    try:
        model_pipe, feature_imp = joblib.load(MODEL_FILE)
    except Exception:
        model_pipe, feature_imp = train_model(df_vis, mode)
        try:
            joblib.dump((model_pipe, feature_imp), MODEL_FILE)
        except Exception:
            pass
else:
    model_pipe, feature_imp = train_model(df_vis, mode)
    if use_joblib:
        try:
            joblib.dump((model_pipe, feature_imp), MODEL_FILE)
        except Exception:
            pass

def predict_property(pipe, inp):
    X_in = pd.DataFrame([inp])
    pred = float(pipe.predict(X_in)[0])
    try:
        rf = pipe.named_steps["rf"]
        pre = pipe.named_steps["pre"]
        Xt = pre.transform(X_in)
        arr = np.stack([t.predict(Xt) for t in rf.estimators_])
        std = float(arr.std())
    except Exception:
        std = 0.0
    def get_annual_rate(data, city):
        try:
            sub = data[data["City"] == city]
            if "Price_per_SqFt" in sub.columns and sub["Price_per_SqFt"].dropna().shape[0] > 10:
                from sklearn.linear_model import LinearRegression
                tmp = sub.dropna(subset=["Price_per_SqFt","Year_Built"])
                lr = LinearRegression()
                lr.fit(tmp[["Year_Built"]], tmp["Price_per_SqFt"])
                slope = lr.coef_[0]
                meanp = tmp["Price_per_SqFt"].mean() if tmp["Price_per_SqFt"].mean() != 0 else 1.0
                rate = slope/meanp
                if np.isnan(rate) or abs(rate)>0.5:
                    return 0.04
                return float(rate)
        except Exception:
            pass
        return 0.04
    annual = get_annual_rate(df_vis, inp["City"])
    future = pred * ((1+annual)**5)
    growth = (future - pred)/pred if pred!=0 else 0
    return {"pred":pred,"std":std,"future":future,"growth":growth,"annual":annual}

# handle predict button
if predict_button:
    input_row = {
        "City": city_in,
        "Property_Type": prop_type_in,
        "BHK": int(bhk_in),
        "Size_in_SqFt": float(size_in),
        "Year_Built": int(year_built),
        "Furnished_Status": furnished,
        "Floor_No": int(floor_no),
        "Total_Floors": int(total_floors)
    }
    res = predict_property(model_pipe, input_row)
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Predicted Current Price (Lakhs)", f"{res['pred']:.2f}")
    col_b.metric("Estimated Price After 5 Years", f"{res['future']:.2f}", delta=f"{(res['future']-res['pred']):.2f} Lakhs")
    col_c.metric("5-year Growth %", f"{res['growth']*100:.1f}%")
    st.write(f"Model uncertainty (std across trees): {res['std']:.2f} Lakhs")
    st.write(f"Annual rate used (approx): {res['annual']*100:.2f}%")

# ---------------------------
# Footer suggestions (uses create_img safely)
# ---------------------------
st.markdown("---")
s1, s2 = st.columns([1,1])
with s1:
    try:
        img = create_img("Design suggestions", ["Color palette & icons to match the UI."])
        st.image(img, use_column_width=True)
    except Exception:
        st.write("Design suggestions image (not available).")
with s2:
    st.write("Legend:")
    st.markdown(f"- KPI color: <span style='color:{KPI}'>orange</span>", unsafe_allow_html=True)
    st.markdown(f"- Headings: <span style='color:{TEXT}'>muted cyan</span>", unsafe_allow_html=True)

st.info("Tip: Switch to 'balanced' or 'accurate' model for production; use joblib caching to avoid retraining.")
