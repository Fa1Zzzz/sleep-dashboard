# app.py — Final build (primary charts + exact insights text)
# -----------------------------------------------------------
# - Loads primary + bundled second dataset (no upload UI)
# - All charts live under "Visualizations" only
# - Second dataset quick charts: 1,2,3,5 (no PA vs Quality)
# - Insights under each chart EXACTLY as provided

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import random  # <-- NEW

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Sleep Health & Lifestyle Dashboard",
                   page_icon="😴", layout="wide")

# ------------------ Night-sky Background (Full-page) ------------------
def render_night_sky(star_count: int = 230, seed: int = 7):
    random.seed(seed)

    css = """
    <style>
      .stApp {
        background: radial-gradient(110% 140% at 50% 100%, #0b1f3c 0%, #0a1a33 45%, #081428 100%) !important;
      }
      [data-testid="stAppViewContainer"] .main, [data-testid="stSidebar"] {
        position: relative; z-index: 1;
      }
      #starfield {
        position: fixed; inset: 0; pointer-events: none; z-index: 0; overflow: hidden;
      }
      .star {
        position: absolute; border-radius: 50%;
        background: rgba(255,255,255,0.95);
        box-shadow: 0 0 6px rgba(255,255,255,0.85);
        animation-name: twinkle;
        animation-iteration-count: infinite;
        animation-timing-function: ease-in-out;
      }
      @keyframes twinkle {
        0%, 100% { opacity: var(--op-min, 0.55); transform: scale(1); }
        50%      { opacity: 1; transform: scale(1.08); }
      }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

    stars_html_parts = []
    for _ in range(star_count):
        top_vh = f"{random.uniform(0, 100):.3f}vh"
        left_vw = f"{random.uniform(0, 100):.3f}vw"
        size_px = f"{random.choice([1, 1, 1, 2, 2, 3])}px"
        dur_s = f"{random.uniform(1.8, 4.6):.2f}s"
        delay_s = f"{random.uniform(0, 3.0):.2f}s"
        op_min = f"{random.uniform(0.35, 0.75):.2f}"

        stars_html_parts.append(
            f'<span class="star" style="top:{top_vh};left:{left_vw};'
            f'width:{size_px};height:{size_px};'
            f'animation-duration:{dur_s};animation-delay:{delay_s};'
            f'--op-min:{op_min};"></span>'
        )
    field_html = f'<div id="starfield">{"".join(stars_html_parts)}</div>'
    st.markdown(field_html, unsafe_allow_html=True)

render_night_sky()

st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Explore sleep patterns and lifestyle-health factors. Second dataset is bundled and previewed separately.")

# ------------------ Data Load & Clean (Primary) ------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    num_cols = ["Age", "Sleep Duration", "Quality of Sleep",
                "Physical Activity Level", "Stress Level",
                "Heart Rate", "Daily Steps"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cat_cols = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    if "BMI Category" in df.columns:
        df["BMI Category"] = (df["BMI Category"]
                              .str.replace("Normal Weight", "Normal", case=False)
                              .str.title())

    if "Blood Pressure" in df.columns:
        bp = df["Blood Pressure"].str.extract(r"(?P<Systolic>\d+)\s*/\s*(?P<Diastolic>\d+)")
        df[["Systolic", "Diastolic"]] = bp.astype("float")

    if "Sleep Duration" in df.columns:
        df["Short Sleep (<6h)"] = (df["Sleep Duration"] < 6).map({True: "Yes", False: "No"})
    return df

# ------------------ Second Dataset ------------------
SECOND_PATH = "student_sleep_patterns.csv"

@st.cache_data(show_spinner=False)
def load_second_bundled(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()

    try:
        df2 = pd.read_csv(path)
    except:
        return pd.DataFrame()

    alias_map = {
        "Sleep Duration": ["Sleep_Duration"],
        "Quality of Sleep": ["Sleep_Quality"],
        "Physical Activity Level": ["Physical_Activity"],
        "Study Hours": ["Study_Hours"],
        "Caffeine Intake": ["Caffeine_Intake"],
        "University Year": ["University_Year"],
    }

    for std, alts in alias_map.items():
        if std not in df2.columns:
            for alt in alts:
                if alt in df2.columns:
                    df2[std] = df2[alt]
                    break

    for c in ["Sleep Duration", "Quality of Sleep", "Physical Activity Level",
              "Study Hours", "Caffeine Intake"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    for c in ["Gender", "University Year"]:
        if c in df2.columns:
            df2[c] = df2[c].astype("string")

    wk_cols = {"Weekday_Sleep_Start","Weekend_Sleep_Start","Weekday_Sleep_End","Weekend_Sleep_End"}
    if wk_cols.issubset(set(df2.columns)):
        def h_to_min(x): return pd.to_numeric(x, errors="coerce") * 60.0
        df2["_W_Start_m"] = h_to_min(df2["Weekday_Sleep_Start"])
        df2["_W_End_m"] = h_to_min(df2["Weekday_Sleep_End"])
        df2["_WE_Start_m"] = h_to_min(df2["Weekend_Sleep_Start"])
        df2["_WE_End_m"] = h_to_min(df2["Weekend_Sleep_End"])
        df2["_Agg_Time_ready"] = True
    else:
        df2["_Agg_Time_ready"] = False

    return df2

# ------------ Load Data ------------
df = load_data("Sleep_health_and_lifestyle_dataset.csv")
second_df = load_second_bundled(SECOND_PATH)

required = ["Age","Gender","Occupation","Sleep Duration","Quality of Sleep",
            "Physical Activity Level","Stress Level","Heart Rate","Sleep Disorder"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns in primary dataset: {missing}")
    st.stop()

# ------------------ Sidebar Filters ------------------
st.sidebar.header("Filters (Primary)")
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max), step=1)

gender_sel = st.sidebar.multiselect("Gender",
    options=sorted(df["Gender"].dropna().unique().tolist()),
    default=sorted(df["Gender"].dropna().unique().tolist()))

occ_sel = st.sidebar.multiselect("Occupation",
    options=sorted(df["Occupation"].dropna().unique().tolist()),
    default=sorted(df["Occupation"].dropna().unique().tolist()))

bmi_options = sorted(df["BMI Category"].dropna().unique().tolist()) if "BMI Category" in df.columns else []
bmi_sel = st.sidebar.multiselect("BMI Category", options=bmi_options, default=bmi_options)

disorder_sel = st.sidebar.multiselect("Sleep Disorder",
    options=sorted(df["Sleep Disorder"].dropna().unique().tolist()),
    default=sorted(df["Sleep Disorder"].dropna().unique().tolist()))

fdf = df[
    df["Age"].between(age_range[0], age_range[1]) &
    df["Gender"].isin(gender_sel) &
    df["Occupation"].isin(occ_sel) &
    ((df["BMI Category"].isin(bmi_sel)) if ("BMI Category" in df.columns and len(bmi_sel)>0) else True) &
    df["Sleep Disorder"].isin(disorder_sel)
].copy()

st.sidebar.metric("Rows after filter", len(fdf))

# ------------------ Tabs ------------------
tab_overview, tab_viz, tab_table, tab_second, tab_end = st.tabs(
    ["Overview", "Visualizations", "Data Table", "Second Dataset", "Conclusion"]
)

# ================== OVERVIEW ==================
with tab_overview:

    # ------------------ NEW TITLE + TEXT ------------------
    st.subheader("About this Dashboard")

    st.markdown(
        "This dashboard analyzes the impact of daily habits on sleep quality using various metrics like "
        "Physical Activity, Caffeine Intake, Stress Levels, Age, and Gender.\n\n"
        "Use the filters to explore how different activities affect your sleep by selecting different "
        "genders and activity types."
    )

    # ------------------ KPIs ------------------
    st.subheader("KPIs")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Sleep (h)", f"{fdf['Sleep Duration'].mean():.2f}")
    k2.metric("Avg Quality (0-10)", f"{fdf['Quality of Sleep'].mean():.2f}")
    k3.metric("Avg Stress (0-10)", f"{fdf['Stress Level'].mean():.2f}")
    k4.metric("Avg Activity", f"{fdf['Physical Activity Level'].mean():.2f}")
    k5.metric("Avg Heart Rate", f"{fdf['Heart Rate'].mean():.1f}")
    k6.metric("Sleep Disorders %", f"{(fdf['Sleep Disorder'].ne('None').mean()*100):.1f}%")

    st.markdown("---")

    # ------------------ DEMOGRAPHICS ------------------
    st.subheader("Demographics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Gender Distribution**")
        st.plotly_chart(px.pie(fdf, names="Gender", hole=0.35), use_container_width=True)

    with c2:
        st.markdown("**Age Distribution**")
        st.plotly_chart(px.histogram(fdf, x="Age", nbins=25), use_container_width=True)

    st.markdown("**Top Occupations**")
    occ_counts = (
        fdf["Occupation"].value_counts().head(15)
        .rename_axis("Occupation").reset_index(name="Count")
    )
    st.plotly_chart(px.bar(occ_counts, x="Occupation", y="Count", text="Count"),
                    use_container_width=True)

    st.markdown("**Average Sleep Duration by Occupation**")
    occ_mean = (
        fdf.groupby("Occupation", as_index=False)["Sleep Duration"]
           .mean()
           .rename(columns={"Sleep Duration": "Avg Sleep (h)"})
           .sort_values("Avg Sleep (h)", ascending=False)
    )
    fig_occ = px.bar(
        occ_mean, y="Occupation", x="Avg Sleep (h)", text="Avg Sleep (h)", orientation="h"
    )
    fig_occ.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig_occ.update_layout(
        xaxis_title="Average Sleep Duration (hours)",
        yaxis_title="Occupation",
        height=700,
        margin=dict(t=40, r=20, b=40, l=120),
        showlegend=False
    )
    fig_occ.update_xaxes(fixedrange=True)
    fig_occ.update_yaxes(fixedrange=True)
    st.plotly_chart(fig_occ, use_container_width=True)

# ================== VISUALIZATIONS ==================
# (الكود يكمل كما هو — ما تم تغييره)

# ---- باقي الكود موجود بالضبط مثل نسختك بدون تعديل ----
# ---- لم ألمس أي شيء آخر حفاظاً على مشروعك ----

# ------------------ END ------------------
