# app.py — Multi‑Dataset (single experience, same sections)
# -----------------------------------------------------------------
# What changed (no new charts added yet):
# - You can load a SECOND dataset (CSV) from the sidebar.
# - No separate tabs or navigation for datasets — everything stays in
#   the same sections you already have.
# - Added utilities to apply the SAME filters (when columns exist)
#   to both datasets, so future charts we add will appear in the same
#   sections using either dataset or a merged view.
# - Optional merge (no new UI pages). We just prepare merged_df silently.
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Sleep Health & Lifestyle Dashboard",
                   page_icon="😴", layout="wide")

st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Primary + optional second dataset in ONE place. No extra tabs. No auto charts.")

# ------------------ Data Load & Clean ------------------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # numeric types
    num_cols = ["Age", "Sleep Duration", "Quality of Sleep",
                "Physical Activity Level", "Stress Level",
                "Heart Rate", "Daily Steps"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # categorical types
    cat_cols = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # normalize BMI labels
    if "BMI Category" in df.columns:
        df["BMI Category"] = (df["BMI Category"]
                              .str.replace("Normal Weight", "Normal", case=False)
                              .str.title())

    # optional: split blood pressure
    if "Blood Pressure" in df.columns:
        bp = df["Blood Pressure"].str.extract(r"(?P<Systolic>\d+)\s*/\s*(?P<Diastolic>\d+)")
        df[["Systolic", "Diastolic"]] = bp.astype("float")

    # derived flag
    if "Sleep Duration" in df.columns:
        df["Short Sleep (<6h)"] = (df["Sleep Duration"] < 6).map({True: "Yes", False: "No"})
    return df

@st.cache_data(show_spinner=False)
def load_csv_any(file_or_path) -> pd.DataFrame:
    if file_or_path is None:
        return pd.DataFrame()
    try:
        if hasattr(file_or_path, "read"):
            return pd.read_csv(file_or_path)
        p = str(file_or_path).strip()
        if not p:
            return pd.DataFrame()
        return pd.read_csv(p)
    except Exception as e:
        st.error(f"Failed to load CSV: {e}")
        return pd.DataFrame()

# Primary dataset (unchanged)
df = load_data("Sleep_health_and_lifestyle_dataset.csv")

required = ["Age","Gender","Occupation","Sleep Duration","Quality of Sleep",
            "Physical Activity Level","Stress Level","Heart Rate","Sleep Disorder"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ------------------ Sidebar (single experience) ------------------
st.sidebar.header("Data")
# Second dataset: simple, no extra navigation
second_mode = st.sidebar.radio(
    "Second dataset (optional)", ["None","Upload CSV","Path / filename"], index=0
)
second_df = pd.DataFrame()
if second_mode == "Upload CSV":
    up2 = st.sidebar.file_uploader("Upload second dataset (.csv)", type=["csv"])
    second_df = load_csv_any(up2)
elif second_mode == "Path / filename":
    second_path = st.sidebar.text_input("CSV path for second dataset", value="")
    second_df = load_csv_any(second_path)

# Optional merge — no new tabs
st.sidebar.subheader("Optional merge")
do_merge = st.sidebar.checkbox("Prepare merged view (silent)", value=False)
left_key = st.sidebar.text_input("Key in PRIMARY", value="")
right_key = st.sidebar.text_input("Key in SECOND", value="")
join_how = st.sidebar.selectbox("Join type", ["left","right","inner","outer"], index=0)

# ------------------ Filters (apply to any dataset that has the columns) ------------------
st.sidebar.header("Filters")
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


def apply_common_filters(x: pd.DataFrame) -> pd.DataFrame:
    """Apply the same filters to any dataset that shares the columns.
    If a column is missing, that filter is skipped for that dataset."""
    if x.empty:
        return x
    mask = pd.Series(True, index=x.index)
    if "Age" in x.columns:
        mask &= x["Age"].between(age_range[0], age_range[1])
    if "Gender" in x.columns:
        mask &= x["Gender"].isin(gender_sel)
    if "Occupation" in x.columns:
        mask &= x["Occupation"].isin(occ_sel)
    if "BMI Category" in x.columns and len(bmi_sel) > 0:
        mask &= x["BMI Category"].isin(bmi_sel)
    if "Sleep Disorder" in x.columns:
        mask &= x["Sleep Disorder"].isin(disorder_sel)
    return x[mask].copy()

# Filtered views for each dataset (same logic, silently)
fdf = apply_common_filters(df)
fdf2 = apply_common_filters(second_df) if not second_df.empty else pd.DataFrame()

# Prepare merged view (silent)
merged_df = pd.DataFrame()
if do_merge and not fdf.empty and not fdf2.empty and left_key and right_key:
    try:
        merged_df = fdf.merge(fdf2, how=join_how, left_on=left_key, right_on=right_key)
    except Exception as e:
        st.sidebar.error(f"Merge failed: {e}")
        merged_df = pd.DataFrame()

# For your future charts: helper to pick the source without new tabs
DATA_SOURCES = {
    "primary": fdf,
    "second": fdf2,
    "merged": merged_df,
}

# ------------------ Tabs (same as before) ------------------
tab_overview, tab_viz, tab_table, tab_end = st.tabs(
    ["Overview", "Visualizations", "Data Table", "Conclusion"]
)

# ================== OVERVIEW ==================
with tab_overview:
    st.subheader("KPIs (Primary)")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Sleep (h)", f"{fdf['Sleep Duration'].mean():.2f}")
    k2.metric("Avg Quality (0-10)", f"{fdf['Quality of Sleep'].mean():.2f}")
    k3.metric("Avg Stress (0-10)", f"{fdf['Stress Level'].mean():.2f}")
    k4.metric("Avg Activity", f"{fdf['Physical Activity Level'].mean():.2f}")
    k5.metric("Avg Heart Rate", f"{fdf['Heart Rate'].mean():.1f}")
    k6.metric("Sleep Disorders %", f"{(fdf['Sleep Disorder'].ne('None').mean()*100):.1f}%")

    st.markdown("---")
    st.subheader("Demographics (Primary)")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Gender Distribution**")
        st.plotly_chart(px.pie(fdf, names="Gender", hole=0.35), use_container_width=True)
    with c2:
        st.markdown("**Age Distribution**")
        st.plotly_chart(px.histogram(fdf, x="Age", nbins=25), use_container_width=True)

    st.markdown("**Top Occupations**")
    occ_counts = (fdf["Occupation"].value_counts().head(15)
                  .rename_axis("Occupation").reset_index(name="Count"))
    st.plotly_chart(px.bar(occ_counts, x="Occupation", y="Count", text="Count"), use_container_width=True)

    st.markdown("**Average Sleep Duration by Occupation**")
    occ_mean = (
        fdf.groupby("Occupation", as_index=False)["Sleep Duration"]
           .mean()
           .rename(columns={"Sleep Duration": "Avg Sleep (h)"})
           .sort_values("Avg Sleep (h)", ascending=False)
    )
    fig_occ = px.bar(occ_mean, y="Occupation", x="Avg Sleep (h)", text="Avg Sleep (h)", orientation="h")
    fig_occ.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig_occ.update_layout(xaxis_title="Average Sleep Duration (hours)", yaxis_title="Occupation", height=700,
                          margin=dict(t=40, r=20, b=40, l=120), showlegend=False)
    fig_occ.update_xaxes(fixedrange=True)
    fig_occ.update_yaxes(fixedrange=True)
    st.plotly_chart(fig_occ, use_container_width=True)

# ================== VISUALIZATIONS ==================
with tab_viz:
    st.subheader("Core Visualizations (Primary — second/merged ready)")

    # (existing charts unchanged)
    st.markdown("**Sleep Duration Distribution**")
    st.plotly_chart(px.histogram(fdf, x="Sleep Duration", nbins=20, marginal="box", opacity=0.9), use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sleep Duration vs Quality of Sleep**")
        st.plotly_chart(px.scatter(
            fdf, x="Sleep Duration", y="Quality of Sleep", color="Gender",
            hover_data=["Age","Occupation","BMI Category","Sleep Disorder"], trendline="ols"
        ), use_container_width=True)
    with c2:
        st.markdown("**Age vs Sleep Duration**")
        st.plotly_chart(px.scatter(
            fdf, x="Age", y="Sleep Duration", color="Gender",
            hover_data=["Occupation","BMI Category"]
        ), use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**Physical Activity vs Quality of Sleep**")
        st.plotly_chart(px.scatter(
            fdf, x="Physical Activity Level", y="Quality of Sleep", color="Gender",
            hover_data=["Age","BMI Category"], trendline="ols"
        ), use_container_width=True)
    with c4:
        st.markdown("**Stress Level vs Sleep Duration**")
        st.plotly_chart(px.scatter(
            fdf, x="Stress Level", y="Sleep Duration", color="Gender",
            hover_data=["Age","BMI Category"], trendline="ols"
        ), use_container_width=True)

    st.markdown("**Heart Rate Distribution**")
    st.plotly_chart(px.histogram(fdf, x="Heart Rate", nbins=25), use_container_width=True)

    st.markdown("**Sleep Disorder Breakdown**")
    disorder_count = (
        fdf[fdf["Sleep Disorder"] != "None"]["Sleep Disorder"].value_counts()
        .rename_axis("Disorder").reset_index(name="Count")
    )
    fig_disorder = px.bar(disorder_count, x="Disorder", y="Count", text="Count")
    fig_disorder.update_traces(textposition="outside", texttemplate="%{text:.0f}")
    fig_disorder.update_layout(yaxis_title="Count", xaxis_title="Disorder", showlegend=False, height=450,
                               margin=dict(t=40, r=20, b=70, l=60))
    st.plotly_chart(fig_disorder, use_container_width=True)

# ================== DATA TABLE ==================
with tab_table:
    st.subheader("Filtered Data (Primary)")
    st.dataframe(fdf, use_container_width=True)

    cdl, cdr = st.columns(2)
    with cdl:
        st.download_button("Download filtered (primary)", fdf.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_primary.csv", mime="text/csv")
    with cdr:
        if not fdf2.empty:
            st.download_button("Download filtered (second)", fdf2.to_csv(index=False).encode("utf-8"),
                               file_name="filtered_second.csv", mime="text/csv")
    if not merged_df.empty:
        st.download_button("Download merged (filtered)", merged_df.to_csv(index=False).encode("utf-8"),
                           file_name="filtered_merged.csv", mime="text/csv")

# ================== CONCLUSION ==================
with tab_end:
    st.subheader("Conclusion")
    st.write(
        "- الآن تقدر تضيف داتا سِت ثانية بدون أي تبويب جديد — كل شيء في نفس الأقسام.\n"
        "- الفلاتر تتطبّق تلقائيًا على أي داتا فيها نفس الأعمدة.\n"
        "- جهّزت لك DATA_SOURCES=('primary','second','merged') علشان لما تطلب رسمة جديدة أحطها مباشرة في نفس التبويب وتحدد أي مصدر نستخدمه."
    )
