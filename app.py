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

    # ------------------------------------------------------------------
    # NEW: User-requested charts added below (use any dataset available)

    # Helper to pick best data source that contains ALL required columns
    def _choose_source(required_cols):
        def has_cols(d, cols):
            return (isinstance(d, pd.DataFrame) and not d.empty and all(c in d.columns for c in cols))
        if has_cols(merged_df, required_cols):
            return merged_df, "merged"
        if has_cols(fdf2, required_cols):
            return fdf2, "second"
        if has_cols(fdf, required_cols):
            return fdf, "primary"
        return pd.DataFrame(), None

    def _union_columns(*dfs):
        cols = set()
        for d in dfs:
            if isinstance(d, pd.DataFrame) and not d.empty:
                cols.update(map(str, d.columns))
        return sorted(cols)

    def _guess(cols_union, *candidates):
        # simple case-insensitive contains/equals match
        lower = {c.lower(): c for c in cols_union}
        for cand in candidates:
            cand_l = cand.lower()
            if cand_l in lower:
                return lower[cand_l]
        # contains
        for c in cols_union:
            cl = c.lower()
            if any(tok in cl for tok in [x.lower() for x in candidates]):
                return c
        return None

    st.markdown("---")
    st.subheader("Additional Visualizations (Requested)")

    # -------- Mapping UI (so your columns don't have to match exact names) --------
    with st.expander("Column Mapping (use if your columns have different names)", expanded=False):
        cols_union = _union_columns(fdf, fdf2, merged_df)
        col_study_hours = st.selectbox(
            "Study Hours column",
            options=["(auto-detect)"] + cols_union,
            index=0,
            help="Select if your Study Hours column has a different name."
        )
        col_univ_year = st.selectbox(
            "University Year column",
            options=["(auto-detect)"] + cols_union,
            index=0
        )
        col_caffeine = st.selectbox(
            "Caffeine Intake column",
            options=["(auto-detect)"] + cols_union,
            index=0
        )
        col_sleep_start = st.selectbox(
            "Sleep Start column",
            options=["(auto-detect)"] + cols_union,
            index=0
        )
        col_sleep_end = st.selectbox(
            "Sleep End column",
            options=["(auto-detect)"] + cols_union,
            index=0
        )
        col_date = st.selectbox(
            "Date column (optional but recommended)",
            options=["(auto-detect)"] + cols_union,
            index=0
        )

    # Autodetect fallbacks if user leaves mapping as auto
    cols_union_all = _union_columns(fdf, fdf2, merged_df)
    if col_study_hours == "(auto-detect)":
        col_study_hours = _guess(cols_union_all, "Study Hours", "StudyHours", "Hours of Study", "study")
    if col_univ_year == "(auto-detect)":
        col_univ_year = _guess(cols_union_all, "University Year", "Year", "Uni Year", "Academic Year")
    if col_caffeine == "(auto-detect)":
        col_caffeine = _guess(cols_union_all, "Caffeine Intake", "Caffeine", "Coffee Cups", "Cups")
    if col_sleep_start == "(auto-detect)":
        col_sleep_start = _guess(cols_union_all, "Sleep Start", "Bedtime", "SleepStart", "Start Time")
    if col_sleep_end == "(auto-detect)":
        col_sleep_end = _guess(cols_union_all, "Sleep End", "Wakeup", "Wake Time", "SleepEnd", "End Time")
    if col_date == "(auto-detect)":
        col_date = _guess(cols_union_all, "Date", "Day", "Record Date", "Datetime")

    # 1) Sleep Duration vs. Study Hours (scatter)
    st.markdown("**Sleep Duration vs Study Hours**")
    req_cols = [c for c in ["Sleep Duration", col_study_hours] if c]
    dsrc, dname = _choose_source(req_cols)
    if dname is None or col_study_hours is None or "Sleep Duration" not in dsrc.columns or col_study_hours not in dsrc.columns:
        st.info("Missing columns for this chart: needs 'Sleep Duration' and a mapped 'Study Hours' column.")
    else:
        tmp = dsrc.copy()
        for c in ["Sleep Duration", col_study_hours]:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        color_col = "Gender" if "Gender" in tmp.columns else None
        fig_sd_sh = px.scatter(
            tmp, x=col_study_hours, y="Sleep Duration", color=color_col,
            trendline="ols", hover_data=[c for c in ["Age","Occupation","BMI Category","Sleep Disorder"] if c in tmp.columns]
        )
        st.plotly_chart(fig_sd_sh, use_container_width=True)

    # 2) Sleep Quality by University Year (box)
    st.markdown("**Sleep Quality by University Year**")
    req_cols = [c for c in ["Quality of Sleep", col_univ_year] if c]
    dsrc, dname = _choose_source(req_cols)
    if dname is None or col_univ_year is None or "Quality of Sleep" not in dsrc.columns or col_univ_year not in dsrc.columns:
        st.info("Missing columns for this chart: needs 'Quality of Sleep' and a mapped 'University Year' column.")
    else:
        tmp = dsrc.copy()
        fig_q_year = px.box(tmp, x=col_univ_year, y="Quality of Sleep", points="outliers")
        st.plotly_chart(fig_q_year, use_container_width=True)

    # 3) Caffeine Intake vs. Sleep Duration (scatter)
    st.markdown("**Caffeine Intake vs Sleep Duration**")
    req_cols = [c for c in [col_caffeine, "Sleep Duration"] if c]
    dsrc, dname = _choose_source(req_cols)
    if dname is None or col_caffeine is None or "Sleep Duration" not in dsrc.columns or col_caffeine not in dsrc.columns:
        st.info("Missing columns for this chart: needs a mapped 'Caffeine Intake' and 'Sleep Duration'.")
    else:
        tmp = dsrc.copy()
        for c in [col_caffeine, "Sleep Duration"]:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        color_col = "Gender" if "Gender" in tmp.columns else None
        fig_caff = px.scatter(tmp, x=col_caffeine, y="Sleep Duration", color=color_col, trendline="ols")
        st.plotly_chart(fig_caff, use_container_width=True)

    # 4) Physical Activity and Sleep Quality — already present above
    st.caption("*Note: 'Physical Activity vs Quality of Sleep' above covers request #4.*")

    # 5) Sleep Start and End Times — line chart, weekdays vs weekends
    st.markdown("**Sleep Start and End Times — Weekdays vs Weekends**")
    needed = [c for c in [col_sleep_start, col_sleep_end] if c]
    # Prefer dataset that also includes 'Date'
    dsrc_date, dname_date = _choose_source(needed + ([col_date] if col_date else []))
    dsrc_only, dname_only = _choose_source(needed)
    dsrc = dsrc_date if dname_date is not None else dsrc_only
    if dsrc is None or dsrc.empty or any(c not in dsrc.columns for c in needed):
        st.info("Missing columns for this chart: map 'Sleep Start' and 'Sleep End' (and optionally 'Date').")
    else:
        tmp = dsrc.copy()
        # Parse to datetime if possible
        for c in [col_sleep_start, col_sleep_end]:
            tmp[c] = pd.to_datetime(tmp[c], errors="coerce")
        if col_date and col_date in tmp.columns:
            tmp[col_date] = pd.to_datetime(tmp[col_date], errors="coerce")
            dow = tmp[col_date].dt.dayofweek
            tmp["Day Type"] = np.where(dow.isin([4,5]), "Weekend", "Weekday")
        elif "Day Type" in tmp.columns:
            tmp["Day Type"] = tmp["Day Type"].astype(str)
        else:
            tmp["Day Type"] = "Unknown"
        tmp = tmp.dropna(subset=[col_sleep_start, col_sleep_end]).copy()
        if tmp.empty or (tmp["Day Type"] == "Unknown").all():
            st.info("Need a Date (or an existing 'Day Type') to separate weekdays vs weekends.")
        else:
            def to_minutes(ts):
                return ts.dt.hour * 60 + ts.dt.minute
            tmp["Start_m"] = to_minutes(tmp[col_sleep_start])
            tmp["End_m"] = to_minutes(tmp[col_sleep_end])
            if col_date and col_date in tmp.columns and tmp[col_date].notna().any():
                plot_df = tmp.dropna(subset=[col_date]).copy()
                plot_df = plot_df.melt(
                    id_vars=[col_date, "Day Type"],
                    value_vars=["Start_m", "End_m"],
                    var_name="Metric", value_name="Minutes"
                )
                plot_df["Metric"] = plot_df["Metric"].map({"Start_m": "Sleep Start", "End_m": "Sleep End"})
                fig_time = px.line(plot_df.sort_values(col_date), x=col_date, y="Minutes",
                                   color="Metric", line_dash="Day Type", hover_data=["Day Type"]) 
                fig_time.update_layout(yaxis_title="Time (minutes since midnight)")
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                agg = tmp.groupby("Day Type", as_index=False)[["Start_m", "End_m"]].mean(numeric_only=True)
                agg = agg.melt(id_vars=["Day Type"], value_vars=["Start_m", "End_m"], var_name="Metric", value_name="Minutes")
                agg["Metric"] = agg["Metric"].map({"Start_m": "Sleep Start", "End_m": "Sleep End"})
                fig_time2 = px.line(agg, x="Day Type", y="Minutes", color="Metric")
                fig_time2.update_layout(yaxis_title="Time (minutes since midnight)")
                st.plotly_chart(fig_time2, use_container_width=True)

# ================== DATA TABLE (Primary) ==================
with tab_table:
    st.subheader("Filtered Data (Primary)")
    st.dataframe(fdf, use_container_width=True)
    st.download_button("Download filtered CSV",
                       fdf.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_sleep_data.csv",
                       mime="text/csv")

# ================== SECOND DATASET (Preview only) ==================
with tab_second:
    st.subheader("Second Dataset — Schema & Preview (no charts yet)")
    if second_df.empty:
        st.info("Upload or specify a path in the sidebar to load the second dataset.")
    else:
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown("**Preview (first 200 rows)**")
            st.dataframe(second_df.head(200), use_container_width=True)
        with c2:
            st.markdown("**Shape**")
            st.write(second_df.shape)
            st.markdown("**Columns & dtypes**")
            st.write(pd.DataFrame({
                'column': second_df.columns,
                'dtype': [str(t) for t in second_df.dtypes]
            }))
        st.download_button(
            "Download second dataset (as-is)",
            second_df.to_csv(index=False).encode("utf-8"),
            file_name="second_dataset.csv",
            mime="text/csv"
        )
        st.caption("When you're ready, tell me which specific charts you want from this dataset and which columns to use.")

# ================== MERGED VIEW (if configured) ==================
with tab_merged:
    st.subheader("Merged View (optional)")
    if merged_df.empty:
        st.info("Enable 'Create merged view' in the sidebar and provide join keys to preview the merge.")
    else:
        st.success(f"Merged rows: {len(merged_df):,}")
        st.dataframe(merged_df.head(200), use_container_width=True)
        st.download_button(
            "Download merged CSV",
            merged_df.to_csv(index=False).encode("utf-8"),
            file_name="merged_view.csv",
            mime="text/csv"
        )

# ================== CONCLUSION ==================
with tab_end:
    st.subheader("Conclusion")
    st.write(
        "- هذه الداشبورد تعرض نظرة عامة على النوم وعلاقته بعوامل نمط الحياة.\n"
        "- تم إضافة تبويب لعرض الداتا سيت الثانية بدون إنشاء رسوم. أخبرني بالرسوم المطلوبة لاحقًا.\n"
        "- يمكن تجهيز دمج اختياري (Merge) إذا وفرت مفاتيح الربط.\n"
        "- قسم Data Table يتيح تنزيل البيانات المفلترة لمزيد من التحليل."
    )
