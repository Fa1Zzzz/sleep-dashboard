# app.py — Two datasets, same sections (no extra sidebar)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# --------------- Page Setup ---------------
st.set_page_config(page_title="Sleep Health & Lifestyle Dashboard",
                   page_icon="😴", layout="wide")
st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Primary + second dataset together in the same sections.")

# --------------- Loaders ---------------
@st.cache_data
def load_primary(path: str) -> pd.DataFrame:
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

@st.cache_data
def load_second(path: str) -> pd.DataFrame:
    # expects columns like: Age, Gender, University_Year, Sleep_Duration,
    # Study_Hours, Caffeine_Intake, Physical_Activity, Sleep_Quality,
    # optional: Sleep_Start, Sleep_End, Date
    try:
        df2 = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    for c in ["Age", "Sleep_Duration", "Study_Hours", "Caffeine_Intake",
              "Physical_Activity", "Sleep_Quality", "Screen_Time"]:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")
    return df2

# --------------- Read data (no extra sidebar UI) ---------------
PRIMARY_PATH = "Sleep_health_and_lifestyle_dataset.csv"
SECOND_PATH  = "student_sleep_patterns.csv"

df = load_primary(PRIMARY_PATH)
df2 = load_second(SECOND_PATH)

required = ["Age","Gender","Occupation","Sleep Duration","Quality of Sleep",
            "Physical Activity Level","Stress Level","Heart Rate","Sleep Disorder"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Primary dataset missing columns: {missing}")
    st.stop()

# --------------- Filters (existing UI فقط) ---------------
st.sidebar.header("Filters")
age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max), step=1)

gender_sel = st.sidebar.multiselect(
    "Gender",
    options=sorted(df["Gender"].dropna().unique().tolist()),
    default=sorted(df["Gender"].dropna().unique().tolist())
)
occ_sel = st.sidebar.multiselect(
    "Occupation",
    options=sorted(df["Occupation"].dropna().unique().tolist()),
    default=sorted(df["Occupation"].dropna().unique().tolist())
)
bmi_opts = sorted(df["BMI Category"].dropna().unique().tolist()) if "BMI Category" in df.columns else []
bmi_sel = st.sidebar.multiselect("BMI Category", options=bmi_opts, default=bmi_opts)
disorder_sel = st.sidebar.multiselect(
    "Sleep Disorder",
    options=sorted(df["Sleep Disorder"].dropna().unique().tolist()),
    default=sorted(df["Sleep Disorder"].dropna().unique().tolist())
)

# apply on primary
fdf = df[
    df["Age"].between(age_range[0], age_range[1]) &
    df["Gender"].isin(gender_sel) &
    df["Occupation"].isin(occ_sel) &
    ((df["BMI Category"].isin(bmi_sel)) if ("BMI Category" in df.columns and len(bmi_sel)>0) else True) &
    df["Sleep Disorder"].isin(disorder_sel)
].copy()

# apply common parts on second (بس اللي موجود)
def apply_common_filters(x: pd.DataFrame) -> pd.DataFrame:
    if x is None or x.empty:
        return pd.DataFrame()
    m = pd.Series(True, index=x.index)
    if "Age" in x.columns:
        m &= x["Age"].between(age_range[0], age_range[1])
    if "Gender" in x.columns:
        m &= x["Gender"].isin(gender_sel)
    return x[m].copy()

fdf2 = apply_common_filters(df2)
st.sidebar.metric("Rows after filter (primary)", len(fdf))
if not fdf2.empty:
    st.sidebar.metric("Rows after filter (second)", len(fdf2))

# --------------- Tabs ---------------
tab_overview, tab_viz, tab_table, tab_end = st.tabs(
    ["Overview", "Visualizations", "Data Table", "Conclusion"]
)

# ================== OVERVIEW ==================
with tab_overview:
    st.subheader("KPIs — Primary")
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    k1.metric("Avg Sleep (h)", f"{fdf['Sleep Duration'].mean():.2f}")
    k2.metric("Avg Quality (0-10)", f"{fdf['Quality of Sleep'].mean():.2f}")
    k3.metric("Avg Stress (0-10)", f"{fdf['Stress Level'].mean():.2f}")
    k4.metric("Avg Activity", f"{fdf['Physical Activity Level'].mean():.2f}")
    k5.metric("Avg Heart Rate", f"{fdf['Heart Rate'].mean():.1f}")
    k6.metric("Sleep Disorders %", f"{(fdf['Sleep Disorder'].ne('None').mean()*100):.1f}%")

    st.markdown("---")
    st.subheader("Demographics — Primary")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Gender Distribution**")
        st.plotly_chart(px.pie(fdf, names="Gender", hole=0.35), use_container_width=True)
    with c2:
        st.markdown("**Age Distribution**")
        st.plotly_chart(px.histogram(fdf, x="Age", nbins=25), use_container_width=True)

    st.markdown("**Average Sleep Duration by Occupation (Primary)**")
    occ_mean = (
        fdf.groupby("Occupation", as_index=False)["Sleep Duration"].mean()
           .rename(columns={"Sleep Duration": "Avg Sleep (h)"})
           .sort_values("Avg Sleep (h)", ascending=False)
    )
    fig_occ = px.bar(occ_mean, y="Occupation", x="Avg Sleep (h)", text="Avg Sleep (h)", orientation="h")
    fig_occ.update_traces(texttemplate="%{text:.2f}", textposition="outside", cliponaxis=False)
    fig_occ.update_layout(
        xaxis_title="Average Sleep Duration (hours)",
        yaxis_title="Occupation",
        height=700, margin=dict(t=40, r=20, b=40, l=120), showlegend=False
    )
    fig_occ.update_xaxes(fixedrange=True)
    fig_occ.update_yaxes(fixedrange=True)
    st.plotly_chart(fig_occ, use_container_width=True)

    if not fdf2.empty:
        st.markdown("---")
        st.subheader("Quick Glance — Second dataset")
        st.write(f"Rows: {len(fdf2):,}")

# ================== VISUALIZATIONS ==================
with tab_viz:
    st.subheader("Core Visualizations — Primary")

    st.markdown("**Sleep Duration Distribution**")
    st.plotly_chart(px.histogram(fdf, x="Sleep Duration", nbins=20, marginal="box", opacity=0.9),
                    use_container_width=True)

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
        st.markdown("**Physical Activity vs Quality of Sleep (Primary)**")
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
        fdf[fdf["Sleep Disorder"] != "None"]["Sleep Disorder"]
        .value_counts().rename_axis("Disorder").reset_index(name="Count")
    )
    st.plotly_chart(px.bar(disorder_count, x="Disorder", y="Count", text="Count"), use_container_width=True)

    # ---- Requested charts (Primary then Second stacked) ----
    def has_all(d, cols):
        return isinstance(d, pd.DataFrame) and not d.empty and all(c in d.columns for c in cols)

    st.markdown("---")
    st.subheader("Requested Visualizations (Primary then Second)")

    # 1) Sleep Duration vs Study Hours
    st.markdown("### 1) Sleep Duration vs Study Hours")
    if has_all(fdf, ["Sleep Duration", "Study Hours"]):
        st.caption("Primary dataset")
        st.plotly_chart(px.scatter(fdf, x="Study Hours", y="Sleep Duration",
                                   color="Gender", trendline="ols"),
                        use_container_width=True)
    if has_all(fdf2, ["Sleep_Duration", "Study_Hours"]):
        st.caption("Second dataset")
        st.plotly_chart(px.scatter(fdf2, x="Study_Hours", y="Sleep_Duration",
                                   color=fdf2["Gender"] if "Gender" in fdf2.columns else None,
                                   trendline="ols"),
                        use_container_width=True)

    # 2) Sleep Quality by University Year
    st.markdown("### 2) Sleep Quality by University Year")
    if has_all(fdf, ["Quality of Sleep", "University Year"]):
        st.caption("Primary dataset")
        st.plotly_chart(px.box(fdf, x="University Year", y="Quality of Sleep", points="outliers"),
                        use_container_width=True)
    if has_all(fdf2, ["Sleep_Quality", "University_Year"]):
        st.caption("Second dataset")
        st.plotly_chart(px.box(fdf2, x="University_Year", y="Sleep_Quality", points="outliers"),
                        use_container_width=True)

    # 3) Caffeine Intake vs Sleep Duration
    st.markdown("### 3) Caffeine Intake vs Sleep Duration")
    if has_all(fdf, ["Caffeine Intake", "Sleep Duration"]):
        st.caption("Primary dataset")
        st.plotly_chart(px.scatter(fdf, x="Caffeine Intake", y="Sleep Duration",
                                   color="Gender", trendline="ols"),
                        use_container_width=True)
    if has_all(fdf2, ["Caffeine_Intake", "Sleep_Duration"]):
        st.caption("Second dataset")
        st.plotly_chart(px.scatter(fdf2, x="Caffeine_Intake", y="Sleep_Duration",
                                   color=fdf2["Gender"] if "Gender" in fdf2.columns else None,
                                   trendline="ols"),
                        use_container_width=True)

    # 4) Physical Activity and Sleep Quality
    st.markdown("### 4) Physical Activity and Sleep Quality")
    if has_all(fdf, ["Physical Activity Level", "Quality of Sleep"]):
        st.caption("Primary dataset")
        st.plotly_chart(px.scatter(fdf, x="Physical Activity Level", y="Quality of Sleep",
                                   color="Gender", trendline="ols"),
                        use_container_width=True)
    if has_all(fdf2, ["Physical_Activity", "Sleep_Quality"]):
        st.caption("Second dataset")
        st.plotly_chart(px.scatter(fdf2, x="Physical_Activity", y="Sleep_Quality",
                                   color=fdf2["Gender"] if "Gender" in fdf2.columns else None,
                                   trendline="ols"),
                        use_container_width=True)

    # 5) Sleep Start and End Times — Weekdays vs Weekends
    st.markdown("### 5) Sleep Start and End Times — Weekdays vs Weekends")
    def render_start_end(df_in, start_col, end_col, date_col=None, label=""):
        if not has_all(df_in, [start_col, end_col]):
            return False
        tmp = df_in.copy()
        for c in [start_col, end_col]:
            tmp[c] = pd.to_datetime(tmp[c], errors="coerce")
        if date_col and date_col in tmp.columns:
            tmp[date_col] = pd.to_datetime(tmp[date_col], errors="coerce")
            dow = tmp[date_col].dt.dayofweek
            tmp["Day Type"] = np.where(dow.isin([4,5]), "Weekend", "Weekday")
        elif "Day Type" in tmp.columns:
            tmp["Day Type"] = tmp["Day Type"].astype(str)
        else:
            tmp["Day Type"] = "Unknown"
        tmp = tmp.dropna(subset=[start_col, end_col]).copy()
        if tmp.empty or (tmp["Day Type"] == "Unknown").all():
            return False
        tmp["Start_m"] = tmp[start_col].dt.hour * 60 + tmp[start_col].dt.minute
        tmp["End_m"]   = tmp[end_col].dt.hour   * 60 + tmp[end_col].dt.minute
        if date_col and date_col in tmp.columns and tmp[date_col].notna().any():
            plot_df = tmp.dropna(subset=[date_col]).copy()
            plot_df = plot_df.melt(id_vars=[date_col, "Day Type"],
                                   value_vars=["Start_m", "End_m"],
                                   var_name="Metric", value_name="Minutes")
            plot_df["Metric"] = plot_df["Metric"].map({"Start_m": "Sleep Start", "End_m": "Sleep End"})
            fig = px.line(plot_df.sort_values(date_col), x=date_col, y="Minutes",
                          color="Metric", line_dash="Day Type")
        else:
            agg = tmp.groupby("Day Type", as_index=False)[["Start_m", "End_m"]].mean(numeric_only=True)
            plot_df = agg.melt(id_vars=["Day Type"],
                               value_vars=["Start_m", "End_m"],
                               var_name="Metric", value_name="Minutes")
            plot_df["Metric"] = plot_df["Metric"].map({"Start_m": "Sleep Start", "End_m": "Sleep End"})
            fig = px.line(plot_df, x="Day Type", y="Minutes", color="Metric")
        st.caption(label)
        st.plotly_chart(fig, use_container_width=True)
        return True

    rendered = False
    rendered |= render_start_end(fdf, "Sleep Start", "Sleep End", date_col="Date", label="Primary dataset")
    _ = render_start_end(fdf2, "Sleep_Start", "Sleep_End", date_col="Date", label="Second dataset")

# ================== DATA TABLE ==================
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
        "- كل الرسومات المطلوبة تتولد تلقائيًا من الداتا المتوفرة داخل نفس التبويب.\n"
        "- لو ودك نضيف KPIs للداتا الثانية أو نحط دمج بمفتاح محدد، علمني الأعمدة وبضيفه."
    )
