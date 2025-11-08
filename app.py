# app.py — Final Version (Primary + Second Dataset + Insights + Conclusion)

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Sleep Health & Lifestyle Dashboard",
                   page_icon="😴", layout="wide")

st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Primary dataset + second dataset. All insights appear under each chart.")

# ------------------ Data Load & Clean (Primary) ------------------
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

# ------------------ Data Load & Clean (Second Dataset) ------------------
SECOND_PATH = "student_sleep_patterns.csv"

@st.cache_data(show_spinner=False)
def load_second_bundled(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df2 = pd.read_csv(path)
    except Exception:
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

    num_cols = [
        "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
        "Study Hours", "Caffeine Intake"
    ]
    for c in num_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    for c in ["Gender", "University Year"]:
        if c in df2.columns:
            df2[c] = df2[c].astype("string")

    wk_cols = {"Weekday_Sleep_Start","Weekend_Sleep_Start","Weekday_Sleep_End","Weekend_Sleep_End"}
    if wk_cols.issubset(set(df2.columns)):
        def h_to_min(x):
            return pd.to_numeric(x, errors="coerce") * 60.0
        df2["_W_Start_m"]   = h_to_min(df2["Weekday_Sleep_Start"])
        df2["_W_End_m"]     = h_to_min(df2["Weekday_Sleep_End"])
        df2["_WE_Start_m"]  = h_to_min(df2["Weekend_Sleep_Start"])
        df2["_WE_End_m"]    = h_to_min(df2["Weekend_Sleep_End"])
        df2["_Agg_Time_ready"] = True
    else:
        df2["_Agg_Time_ready"] = False

    return df2


# ------------ Load datasets ------------
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

# Apply filters
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
    st.subheader("KPIs")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Sleep (h)", f"{fdf['Sleep Duration'].mean():.2f}")
    k2.metric("Avg Quality (0-10)", f"{fdf['Quality of Sleep'].mean():.2f}")
    k3.metric("Avg Stress (0-10)", f"{fdf['Stress Level'].mean():.2f}")
    k4.metric("Avg Activity", f"{fdf['Physical Activity Level'].mean():.2f}")
    k5.metric("Avg Heart Rate", f"{fdf['Heart Rate'].mean():.1f}")
    k6.metric("Sleep Disorders %", f"{(fdf['Sleep Disorder'].ne('None').mean()*100):.1f}%")

# ================== VISUALIZATIONS ==================
with tab_viz:
    st.subheader("Core Visualizations (Primary Dataset)")

    # ------------------ PRIMARY CHARTS + INSIGHTS ------------------

    # 1) Sleep Duration Histogram
    st.markdown("### Sleep Duration Distribution")
    fig1 = px.histogram(fdf, x="Sleep Duration", nbins=20, marginal="box", opacity=0.9)
    st.plotly_chart(fig1, use_container_width=True)
    st.markdown("**Insight:** This chart displays a boxplot of sleep durations, where the median sleep duration is around 6.5 hours. The “minimum value” indicates 5.8 hours of sleep, suggesting that some individuals might have shorter sleep durations.")

    # 2) Sleep Duration vs Quality
    st.markdown("### Sleep Duration vs Quality of Sleep")
    fig2 = px.scatter(
        fdf, x="Sleep Duration", y="Quality of Sleep",
        color="Gender",
        hover_data=[c for c in ["Age","Occupation","BMI Category","Sleep Disorder"] if c in fdf.columns],
        trendline="ols"
    )
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown("**Insight:** This scatter plot demonstrates a positive correlation between sleep duration and quality of sleep. As sleep duration increases, so does sleep quality. Females generally show slightly higher sleep quality compared to males.")

    # 3) Age vs Sleep
    st.markdown("### Age vs Sleep Duration")
    fig3 = px.scatter(
        fdf, x="Age", y="Sleep Duration",
        color="Gender",
        hover_data=[c for c in ["Occupation","BMI Category"] if c in fdf.columns]
    )
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown("**Insight:** Sleep duration does not significantly vary with age. The data points show no clear trend, suggesting that other factors influence sleep duration more strongly than age.")

    # 4) Physical Activity vs Sleep Quality (PRIMARY ONLY)
    st.markdown("### Physical Activity vs Sleep Quality")
    fig4 = px.scatter(
        fdf, x="Physical Activity Level", y="Quality of Sleep",
        color="Gender",
        hover_data=[c for c in ["Age","BMI Category"] if c in fdf.columns],
        trendline="ols"
    )
    st.plotly_chart(fig4, use_container_width=True)
    st.markdown(
        "**Insight:** The data shows a positive relationship between physical activity levels and sleep quality. "
        "As physical activity increases, the quality of sleep improves. The effect appears more pronounced for males."
    )

    # 5) Stress vs Sleep
    st.markdown("### Stress Level vs Sleep Duration")
    fig5 = px.scatter(
        fdf, x="Stress Level", y="Sleep Duration",
        color="Gender",
        hover_data=[c for c in ["Age","BMI Category"] if c in fdf.columns],
        trendline="ols"
    )
    st.plotly_chart(fig5, use_container_width=True)
    st.markdown("**Insight:** Higher stress levels are associated with reduced sleep duration, with the effect more noticeable among males.")

    # 6) Heart Rate Distribution
    st.markdown("### Heart Rate Distribution")
    fig6 = px.histogram(fdf, x="Heart Rate", nbins=25)
    st.plotly_chart(fig6, use_container_width=True)
    st.markdown("**Insight:** Most individuals have heart rates between 65 and 75 bpm, indicating a generally normal resting heart rate range.")

    st.markdown("---")
    st.subheader("Requested Quick Charts (Second Dataset)")

    # ------------------ SECOND DATASET CHARTS (minus removed one) ------------------

    def _has(df, cols):
        return isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in cols)

    if second_df.empty:
        st.warning("⚠️ Second dataset not found.")
    else:

        # 1) Sleep Duration vs Study Hours
        st.markdown("### Sleep Duration vs Study Hours")
        if _has(second_df, ["Sleep Duration", "Study Hours"]):
            tmp = second_df.copy()
            fig_sd = px.scatter(tmp, x="Study Hours", y="Sleep Duration", color="Gender", trendline="ols")
            st.plotly_chart(fig_sd, use_container_width=True)
        st.markdown("**Insight:** There is a weak negative correlation between study hours and sleep duration. More study hours slightly reduce sleep duration.")

        # 2) Sleep Quality by University Year
        st.markdown("### Sleep Quality by University Year")
        if _has(second_df, ["Quality of Sleep", "University Year"]):
            tmp = second_df.copy()
            fig_q = px.box(tmp, x="University Year", y="Quality of Sleep")
            st.plotly_chart(fig_q, use_container_width=True)
        st.markdown("**Insight:** Sleep quality remains fairly consistent across all university years with no major differences.")

        # ❌ REMOVED PHYSICAL ACTIVITY VS SLEEP QUALITY (SECOND DATASET)

        # 3) Caffeine Intake vs Sleep Duration
        st.markdown("### Caffeine Intake vs Sleep Duration")
        if _has(second_df, ["Caffeine Intake", "Sleep Duration"]):
            tmp = second_df.copy()
            fig_c = px.scatter(tmp, x="Caffeine Intake", y="Sleep Duration", color="Gender", trendline="ols")
            st.plotly_chart(fig_c, use_container_width=True)
        st.markdown("**Insight:** Caffeine intake affects females more negatively by reducing sleep duration, while males are less impacted.")

        # 4) Sleep Start / End — Weekday vs Weekend
        st.markdown("### Sleep Start and End Times — Weekdays vs Weekends")
        wk_cols = {"Weekday_Sleep_Start","Weekend_Sleep_Start","Weekday_Sleep_End","Weekend_Sleep_End"}
        if wk_cols.issubset(set(second_df.columns)) and bool(second_df["_Agg_Time_ready"].iloc[0]):
            tmp = second_df.copy()

            agg = pd.DataFrame({
                "Day Type": ["Weekday","Weekend","Weekday","Weekend"],
                "Metric":  ["Sleep Start","Sleep Start","Sleep End","Sleep End"],
                "Minutes": [
                    tmp["_W_Start_m"].mean(skipna=True),
                    tmp["_WE_Start_m"].mean(skipna=True),
                    tmp["_W_End_m"].mean(skipna=True),
                    tmp["_WE_End_m"].mean(skipna=True),
                ]
            })
            fig_time = px.line(agg, x="Day Type", y="Minutes", color="Metric")
            st.plotly_chart(fig_time, use_container_width=True)

        st.markdown(
            "**Insight:** On weekends, people tend to sleep earlier and wake up later compared to weekdays. "
            "Weekdays show later sleep start times and earlier wake times."
        )


# ================== DATA TABLE ==================
with tab_table:
    st.subheader("Filtered Data (Primary)")
    st.dataframe(fdf, use_container_width=True)
    st.download_button("Download filtered CSV",
                       fdf.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_sleep_data.csv",
                       mime="text/csv")


# ================== SECOND DATASET PREVIEW ==================
with tab_second:
    st.subheader("Second Dataset — Preview Only")
    if second_df.empty:
        st.warning("Second dataset not found.")
    else:
        st.dataframe(second_df.head(200), use_container_width=True)


# ================== CONCLUSION ==================
with tab_end:
    st.subheader("Conclusion")
    st.markdown("""
### Key Conclusions:
- **Study Hours and Sleep Duration:** A weak negative correlation exists. More study hours slightly reduce sleep duration.  
- **Caffeine Intake and Sleep Duration:** Females lose more sleep as caffeine intake increases compared to males.  
- **Physical Activity and Sleep Quality:** Physical activity improves sleep quality, especially for females.  
- **Age and Sleep Duration:** No clear correlation between age and sleep duration.  
- **Stress Level and Sleep Duration:** Higher stress levels are linked to shorter sleep durations.  

### Recommendations:
1. Promote better time management to help students balance study and sleep.  
2. Educate students—especially females—on caffeine’s impact on sleep.  
3. Encourage regular physical activity to enhance sleep quality.  
4. Implement stress-management programs to improve sleep duration.  
    """)

