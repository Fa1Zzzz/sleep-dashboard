# app.py — Multi‑Dataset Ready (no new charts yet)
# -------------------------------------------------------------
# Your original dashboard kept intact. This upgrade only:
# 1) Lets you add a SECOND dataset (upload or path)
# 2) Shows schema + preview for the second dataset
# 3) (Optional) Prepares a merged view if you provide join keys
# 4) Adds tabs for Second Dataset & Merged View — no charts added yet
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Sleep Health & Lifestyle Dashboard",
                   page_icon="😴", layout="wide")

st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Explore sleep patterns and lifestyle-health factors interactively. Now supports a second dataset (no auto charts).")

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

    # normalize BMI labels (even if BMI chart is removed)
    if "BMI Category" in df.columns:
        df["BMI Category"] = (df["BMI Category"]
                              .str.replace("Normal Weight", "Normal", case=False)
                              .str.title())

    # optional: split blood pressure (chart removed but parsing is safe)
    if "Blood Pressure" in df.columns:
        bp = df["Blood Pressure"].str.extract(r"(?P<Systolic>\d+)\s*/\s*(?P<Diastolic>\d+)")
        df[["Systolic", "Diastolic"]] = bp.astype("float")

    # derived flag
    if "Sleep Duration" in df.columns:
        df["Short Sleep (<6h)"] = (df["Sleep Duration"] < 6).map({True: "Yes", False: "No"})
    return df

@st.cache_data(show_spinner=False)
def load_uploaded_csv(file) -> pd.DataFrame:
    """Safe loader for an uploaded CSV (second dataset)."""
    if file is None:
        return pd.DataFrame()
    try:
        df2 = pd.read_csv(file)
    except Exception as e:
        st.error(f"Failed to read uploaded CSV: {e}")
        return pd.DataFrame()
    return df2

# Primary dataset (unchanged)
df = load_data("Sleep_health_and_lifestyle_dataset.csv")

required = ["Age","Gender","Occupation","Sleep Duration","Quality of Sleep",
            "Physical Activity Level","Stress Level","Heart Rate","Sleep Disorder"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ------------------ Sidebar Filters (Primary) ------------------
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

# Apply filters (Primary)
fdf = df[
    df["Age"].between(age_range[0], age_range[1]) &
    df["Gender"].isin(gender_sel) &
    df["Occupation"].isin(occ_sel) &
    ((df["BMI Category"].isin(bmi_sel)) if ("BMI Category" in df.columns and len(bmi_sel)>0) else True) &
    df["Sleep Disorder"].isin(disorder_sel)
].copy()

st.sidebar.metric("Rows after filter", len(fdf))

# ------------------ Sidebar: Second Dataset Controls ------------------
st.sidebar.markdown("---")
st.sidebar.header("Second Dataset (NEW)")
second_src = st.sidebar.radio(
    "Provide second dataset as:",
    options=["Upload CSV", "Path / filename"],
    index=0,
)

second_df = pd.DataFrame()
if second_src == "Upload CSV":
    up = st.sidebar.file_uploader("Upload second dataset (.csv)", type=["csv"], accept_multiple_files=False)
    second_df = load_uploaded_csv(up)
else:
    second_path = st.sidebar.text_input("CSV path for second dataset", value="")
    if second_path:
        # minimal safe loader without cleaning (you'll request charts/cleaning later)
        try:
            second_df = pd.read_csv(second_path)
        except Exception as e:
            st.sidebar.error(f"Failed to read: {e}")
            second_df = pd.DataFrame()

# Optional merge settings
st.sidebar.subheader("Optional: Merge with Primary")
do_merge = st.sidebar.checkbox("Create merged view", value=False, help="Provide join keys to preview a merged DataFrame.")
left_key = st.sidebar.text_input("Key in PRIMARY", value="")
right_key = st.sidebar.text_input("Key in SECOND", value="")
join_how = st.sidebar.selectbox("Join type", ["left","right","inner","outer"], index=0)

# Prepare merged view (no extra cleaning to avoid assumptions)
merged_df = pd.DataFrame()
if do_merge and not fdf.empty and not second_df.empty and left_key and right_key:
    try:
        merged_df = fdf.merge(second_df, how=join_how, left_on=left_key, right_on=right_key)
    except Exception as e:
        st.sidebar.error(f"Merge failed: {e}")
        merged_df = pd.DataFrame()

# ------------------ Tabs ------------------
tab_overview, tab_viz, tab_table, tab_second, tab_merged, tab_end = st.tabs(
    ["Overview", "Visualizations", "Data Table", "Second Dataset", "Merged View", "Conclusion"]
)

# ================== OVERVIEW ==================
with tab_overview:
    st.subheader("KPIs")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Sleep (h)", f"{fdf['Sleep Duration'].mean():.2f}")
    k2.metric("Avg Quality (0-10)", f"{fdf['Quality of Sleep'].mean():.2f}")
    k3.metric("Avg Stress (0-10)", f"{fdf['Stress Level'].mean():.2f}")
    k4.metric("Avg Activity", f"{fdf['Physical Activity Level'].mean():.2f}")
    k5.metric("Avg Heart Rate", f"{fdf['Heart Rate'].mean():.1f}")
    k6.metric("Sleep Disorders %", f"{(fdf['Sleep Disorder'].ne('None').mean()*100):.1f}%")

    st.markdown("---")
    st.subheader("Demographics")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Gender Distribution**")
        st.plotly_chart(px.pie(fdf, names="Gender", hole=0.35),
                        use_container_width=True)
    with c2:
        st.markdown("**Age Distribution**")
        st.plotly_chart(px.histogram(fdf, x="Age", nbins=25),
                        use_container_width=True)

    st.markdown("**Top Occupations**")
    occ_counts = (fdf["Occupation"].value_counts().head(15)
                  .rename_axis("Occupation").reset_index(name="Count"))
    st.plotly_chart(px.bar(occ_counts, x="Occupation", y="Count", text="Count"),
                    use_container_width=True)

    # Average Sleep Duration by Occupation
    st.markdown("**Average Sleep Duration by Occupation**")
    occ_mean = (
        fdf.groupby("Occupation", as_index=False)["Sleep Duration"]
           .mean()
           .rename(columns={"Sleep Duration": "Avg Sleep (h)"})
           .sort_values("Avg Sleep (h)", ascending=False)
    )
    fig_occ = px.bar(
        occ_mean, y="Occupation", x="Avg Sleep (h)", text="Avg Sleep (h)",
        orientation="h"
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

# ================== VISUALIZATIONS (Primary only) ==================
with tab_viz:
    st.subheader("Core Visualizations (Primary Dataset)")

    # 1) Sleep Duration Distribution
    st.markdown("**Sleep Duration Distribution**")
    fig1 = px.histogram(fdf, x="Sleep Duration", nbins=20, marginal="box", opacity=0.9)
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        # 2) Sleep Duration vs Quality of Sleep
        st.markdown("**Sleep Duration vs Quality of Sleep**")
        fig2 = px.scatter(
            fdf, x="Sleep Duration", y="Quality of Sleep",
            color="Gender",
            hover_data=["Age","Occupation","BMI Category","Sleep Disorder"],
            trendline="ols"
        )
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        # 3) Age vs Sleep Duration
        st.markdown("**Age vs Sleep Duration**")
        fig3 = px.scatter(
            fdf, x="Age", y="Sleep Duration",
            color="Gender",
            hover_data=["Occupation","BMI Category"]
        )
        st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        # 4) Physical Activity vs Quality (colored by Gender)
        st.markdown("**Physical Activity vs Quality of Sleep**")
        fig4 = px.scatter(
            fdf, x="Physical Activity Level", y="Quality of Sleep",
            color="Gender",
            hover_data=["Age","BMI Category"], trendline="ols"
        )
        st.plotly_chart(fig4, use_container_width=True)
    with c4:
        # 5) Stress vs Sleep Duration (colored by Gender)
        st.markdown("**Stress Level vs Sleep Duration**")
        fig5 = px.scatter(
            fdf, x="Stress Level", y="Sleep Duration",
            color="Gender",
            hover_data=["Age","BMI Category"], trendline="ols"
        )
        st.plotly_chart(fig5, use_container_width=True)

    # 6) Heart Rate Distribution
    st.markdown("**Heart Rate Distribution**")
    fig6 = px.histogram(fdf, x="Heart Rate", nbins=25)
    st.plotly_chart(fig6, use_container_width=True)

    # Sleep Disorder Breakdown
    st.markdown("**Sleep Disorder Breakdown**")
    disorder_count = (
        fdf[fdf["Sleep Disorder"] != "None"]["Sleep Disorder"]
        .value_counts()
        .rename_axis("Disorder").reset_index(name="Count")
    )
    fig_disorder = px.bar(
        disorder_count, x="Disorder", y="Count", text="Count"
    )
    fig_disorder.update_traces(textposition="outside", texttemplate="%{text:.0f}")
    fig_disorder.update_layout(
        yaxis_title="Count",
        xaxis_title="Disorder",
        showlegend=False,
        height=450,
        margin=dict(t=40, r=20, b=70, l=60)
    )
    st.plotly_chart(fig_disorder, use_container_width=True)
    # ---------------- YOUR REQUESTED 5 CHARTS -----------------
    st.markdown("---")
    st.subheader("Requested Visualizations — shown for each dataset (no extra options)")

    # Column maps per dataset
    primary_map = {
        "sleep_duration": "Sleep Duration",
        "study_hours": "Study Hours",
        "sleep_quality": "Quality of Sleep",
        "university_year": "University Year",
        "caffeine": "Caffeine Intake",
        "physical_activity": "Physical Activity Level",
        "gender": "Gender",
        "sleep_start": "Sleep Start",
        "sleep_end": "Sleep End",
        "date": "Date",
    }
    second_map = {
        "sleep_duration": "Sleep_Duration",
        "study_hours": "Study_Hours",
        "sleep_quality": "Sleep_Quality",
        "university_year": "University_Year",
        "caffeine": "Caffeine_Intake",
        "physical_activity": "Physical_Activity",
        "gender": "Gender",
        "sleep_start": "Sleep_Start",
        "sleep_end": "Sleep_End",
        "date": "Date",
    }

    def has_all(d, cols):
        return isinstance(d, pd.DataFrame) and not d.empty and all(c in d.columns for c in cols)

    def render_block(title, d, m):
        exp = st.expander(title, expanded=True)
        with exp:
            # 1) Sleep Duration vs Study Hours (scatter)
            st.markdown("**1) Sleep Duration vs Study Hours (scatter)**")
            if has_all(d, [m["sleep_duration"], m["study_hours"]]):
                st.plotly_chart(
                    px.scatter(
                        d, x=m["study_hours"], y=m["sleep_duration"],
                        color=m["gender"] if m["gender"] in d.columns else None,
                        trendline="ols"
                    ), use_container_width=True
                )
            else:
                st.info(f"Needs columns: {m['sleep_duration']} + {m['study_hours']}")

            # 2) Sleep Quality by University Year (box)
            st.markdown("**2) Sleep Quality by University Year (box)**")
            if has_all(d, [m["sleep_quality"], m["university_year"]]):
                st.plotly_chart(
                    px.box(d, x=m["university_year"], y=m["sleep_quality"], points="outliers"),
                    use_container_width=True
                )
            else:
                st.info(f"Needs columns: {m['sleep_quality']} + {m['university_year']}")

            # 3) Caffeine Intake vs Sleep Duration (scatter)
            st.markdown("**3) Caffeine Intake vs Sleep Duration (scatter)**")
            if has_all(d, [m["caffeine"], m["sleep_duration"]]):
                st.plotly_chart(
                    px.scatter(
                        d, x=m["caffeine"], y=m["sleep_duration"],
                        color=m["gender"] if m["gender"] in d.columns else None,
                        trendline="ols"
                    ), use_container_width=True
                )
            else:
                st.info(f"Needs columns: {m['caffeine']} + {m['sleep_duration']}")

            # 4) Physical Activity and Sleep Quality (scatter)
            st.markdown("**4) Physical Activity and Sleep Quality (scatter)**")
            if has_all(d, [m["physical_activity"], m["sleep_quality"]]):
                st.plotly_chart(
                    px.scatter(
                        d, x=m["physical_activity"], y=m["sleep_quality"],
                        color=m["gender"] if m["gender"] in d.columns else None,
                        trendline="ols"
                    ), use_container_width=True
                )
            else:
                st.info(f"Needs columns: {m['physical_activity']} + {m['sleep_quality']}")

            # 5) Sleep Start and End Times — Weekdays vs Weekends (line)
            st.markdown("**5) Sleep Start and End Times — Weekdays vs Weekends (line)**")
            start_c, end_c = m["sleep_start"], m["sleep_end"]
            if has_all(d, [start_c, end_c]) and (m["date"] in d.columns or "Day Type" in d.columns):
                tmp = d.copy()
                tmp[start_c] = pd.to_datetime(tmp[start_c], errors="coerce")
                tmp[end_c]   = pd.to_datetime(tmp[end_c], errors="coerce")
                if m["date"] in tmp.columns:
                    tmp[m["date"]] = pd.to_datetime(tmp[m["date"]], errors="coerce")
                    dow = tmp[m["date"]].dt.dayofweek
                    tmp["Day Type"] = np.where(dow.isin([4,5]), "Weekend", "Weekday")
                tmp = tmp.dropna(subset=[start_c, end_c]).copy()
                tmp["Start_m"] = tmp[start_c].dt.hour*60 + tmp[start_c].dt.minute
                tmp["End_m"]   = tmp[end_c].dt.hour*60 + tmp[end_c].dt.minute
                if m["date"] in tmp.columns and tmp[m["date"]].notna().any():
                    plot_df = tmp.dropna(subset=[m["date"]]).copy()
                    melt = plot_df.melt(id_vars=[m["date"], "Day Type"], value_vars=["Start_m","End_m"],
                                        var_name="Metric", value_name="Minutes")
                    melt["Metric"] = melt["Metric"].map({"Start_m":"Sleep Start","End_m":"Sleep End"})
                    fig = px.line(melt.sort_values(m["date"]), x=m["date"], y="Minutes", color="Metric", line_dash="Day Type")
                else:
                    agg = tmp.groupby("Day Type", as_index=False)[["Start_m","End_m"]].mean(numeric_only=True)
                    melt = agg.melt(id_vars=["Day Type"], value_vars=["Start_m","End_m"], var_name="Metric", value_name="Minutes")
                    melt["Metric"] = melt["Metric"].map({"Start_m":"Sleep Start","End_m":"Sleep End"})
                    fig = px.line(melt, x="Day Type", y="Minutes", color="Metric")
                fig.update_layout(yaxis_title="Minutes since midnight")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Needs Sleep_Start/Sleep_End (and Date preferred) to split weekdays vs weekends.")

    # Render per dataset
    render_block("Primary — Sleep_health_and_lifestyle_dataset", fdf, primary_map)
    if not fdf2.empty:
        render_block("Second — student_sleep_patterns.csv", fdf2, second_map)

# ================== DATA TABLE (Primary) ==================
 (Primary) ==================
with tab_table:
    st.subheader("Filtered Data (Primary)")
    st.dataframe(fdf, use_container_width=True)
    st.download_button("Download filtered CSV",
                       fdf.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_sleep_data.csv",
                       mime="text/csv")

# ================== CONCLUSION ==================

with tab_end:
    st.subheader("Conclusion")
    st.write(
        "- هذه الداشبورد تعرض نظرة عامة على النوم وعلاقته بعوامل نمط الحياة.\n"
        "- تم إضافة تبويب لعرض الداتا سيت الثانية بدون إنشاء رسوم. أخبرني بالرسوم المطلوبة لاحقًا.\n"
        "- يمكن تجهيز دمج اختياري (Merge) إذا وفرت مفاتيح الربط.\n"
        "- قسم Data Table يتيح تنزيل البيانات المفلترة لمزيد من التحليل."
    )
