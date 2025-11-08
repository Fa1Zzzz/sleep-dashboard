# app.py — Multi-Dataset (second CSV is built-in, same sections)
# -----------------------------------------------------------------
# What's included:
# - Loads primary CSV: "Sleep_health_and_lifestyle_dataset.csv"
# - Loads second   CSV: "student_sleep_patterns.csv"  (auto, no upload)
# - Same sections (Overview / Visualizations / Data Table / Conclusion)
# - Preview tab for the second dataset (optional)
# - Optional merge (from sidebar) if you specify join keys
# - Robust column-mapping for your new requested charts
# -----------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Sleep Health & Lifestyle Dashboard",
                   page_icon="😴", layout="wide")

st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Primary + built-in second dataset, all in the same sections. No uploads needed.")

# ------------------ Data Load & Clean ------------------
@st.cache_data
def load_primary(path: str) -> pd.DataFrame:
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

@st.cache_data
def load_second_builtin(path: str) -> pd.DataFrame:
    """Load the built-in second dataset (no cleaning assumptions)."""
    try:
        df2 = pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to read second dataset '{path}': {e}")
        return pd.DataFrame()
    return df2

# Load datasets (from same folder)
PRIMARY_CSV = "Sleep_health_and_lifestyle_dataset.csv"
SECOND_CSV  = "student_sleep_patterns.csv"

df  = load_primary(PRIMARY_CSV)
df2 = load_second_builtin(SECOND_CSV)

required = ["Age","Gender","Occupation","Sleep Duration","Quality of Sleep",
            "Physical Activity Level","Stress Level","Heart Rate","Sleep Disorder"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"Missing columns in primary dataset: {missing}")
    st.stop()

# ------------------ Sidebar Filters (apply to datasets that have these cols) ------------------
st.sidebar.header("Filters")

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", age_min, age_max, (age_min, age_max), step=1)

gender_sel = st.sidebar.multiselect(
    "Gender",
    options=sorted(df["Gender"].dropna().unique().tolist()),
    default=sorted(df["Gender"].dropna().unique().tolist()),
)

occ_sel = st.sidebar.multiselect(
    "Occupation",
    options=sorted(df["Occupation"].dropna().unique().tolist()),
    default=sorted(df["Occupation"].dropna().unique().tolist()),
)

bmi_options = sorted(df["BMI Category"].dropna().unique().tolist()) if "BMI Category" in df.columns else []
bmi_sel = st.sidebar.multiselect("BMI Category", options=bmi_options, default=bmi_options)

disorder_sel = st.sidebar.multiselect(
    "Sleep Disorder",
    options=sorted(df["Sleep Disorder"].dropna().unique().tolist()),
    default=sorted(df["Sleep Disorder"].dropna().unique().tolist()),
)

def apply_common_filters(x: pd.DataFrame) -> pd.DataFrame:
    """Apply same filters to any dataset that has the columns; skip missing cols."""
    if not isinstance(x, pd.DataFrame) or x.empty:
        return pd.DataFrame()
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

fdf  = apply_common_filters(df)          # filtered primary
fdf2 = apply_common_filters(df2)         # filtered second

# ------------------ Optional merge (silent) ------------------
st.sidebar.subheader("Optional merge")
do_merge = st.sidebar.checkbox("Prepare merged view", value=False, help="Provide join keys to preview merged_df in charts.")
left_key  = st.sidebar.text_input("Key in PRIMARY", value="")
right_key = st.sidebar.text_input("Key in SECOND",  value="")
join_how  = st.sidebar.selectbox("Join type", ["left","right","inner","outer"], index=0)

merged_df = pd.DataFrame()
if do_merge and not fdf.empty and not fdf2.empty and left_key and right_key:
    try:
        merged_df = fdf.merge(fdf2, how=join_how, left_on=left_key, right_on=right_key)
    except Exception as e:
        st.sidebar.error(f"Merge failed: {e}")
        merged_df = pd.DataFrame()

st.sidebar.metric("Rows after filter (primary)", len(fdf))
if not fdf2.empty:
    st.sidebar.metric("Rows after filter (second)", len(fdf2))
if not merged_df.empty:
    st.sidebar.metric("Rows after filter (merged)", len(merged_df))

# ------------------ Tabs ------------------
tab_overview, tab_viz, tab_table, tab_second, tab_end = st.tabs(
    ["Overview", "Visualizations", "Data Table", "Second Dataset", "Conclusion"]
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
with tab_viz:
    st.subheader("Core Visualizations (Primary)")

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
        fdf[fdf["Sleep Disorder"] != "None"]["Sleep Disorder"]
        .value_counts().rename_axis("Disorder").reset_index(name="Count")
    )
    fig_disorder = px.bar(disorder_count, x="Disorder", y="Count", text="Count")
    fig_disorder.update_traces(textposition="outside", texttemplate="%{text:.0f}")
    fig_disorder.update_layout(yaxis_title="Count", xaxis_title="Disorder",
                               showlegend=False, height=450,
                               margin=dict(t=40, r=20, b=70, l=60))
    st.plotly_chart(fig_disorder, use_container_width=True)

    # ------------------------------------------------------------------
    # NEW: Your requested charts (works with merged > second > primary)

    # Robust helpers
    def _df_or_empty(obj): return obj if isinstance(obj, pd.DataFrame) else pd.DataFrame()

    _f_primary = _df_or_empty(fdf)
    _f_second  = _df_or_empty(fdf2)
    _f_merged  = _df_or_empty(merged_df)

    def _has_cols(d, cols):
        return (isinstance(d, pd.DataFrame) and not d.empty and all(c in d.columns for c in cols))

    def _choose_source(required_cols):
        if _has_cols(_f_merged, required_cols):  return _f_merged, 'merged'
        if _has_cols(_f_second, required_cols):  return _f_second, 'second'
        if _has_cols(_f_primary, required_cols): return _f_primary, 'primary'
        return pd.DataFrame(), None

    def _union_columns(*dfs):
        cols = set()
        for d in dfs:
            if isinstance(d, pd.DataFrame) and not d.empty:
                cols.update(map(str, d.columns))
        return sorted(cols)

    def _guess(cols_union, *cands):
        lower = {c.lower(): c for c in cols_union}
        for cand in cands:
            if cand.lower() in lower:
                return lower[cand.lower()]
        for c in cols_union:
            cl = c.lower()
            if any(tok in cl for tok in [x.lower() for x in cands]):
                return c
        return None

    st.markdown("---")
    st.subheader("Additional Visualizations (Requested)")

    # Column Mapping (use if your columns have different names)
    with st.expander("Column Mapping (use if your columns have different names)", expanded=False):
        cols_union = _union_columns(_f_primary, _f_second, _f_merged)
        col_study_hours = st.selectbox("Study Hours column", ["(auto-detect)"] + cols_union, index=0)
        col_univ_year   = st.selectbox("University Year column", ["(auto-detect)"] + cols_union, index=0)
        col_caffeine    = st.selectbox("Caffeine Intake column", ["(auto-detect)"] + cols_union, index=0)
        col_sleep_start = st.selectbox("Sleep Start column", ["(auto-detect)"] + cols_union, index=0)
        col_sleep_end   = st.selectbox("Sleep End column", ["(auto-detect)"] + cols_union, index=0)
        col_date        = st.selectbox("Date column (optional)", ["(auto-detect)"] + cols_union, index=0)

    cols_union_all = _union_columns(_f_primary, _f_second, _f_merged)
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

    # 1) Sleep Duration vs Study Hours (scatter)
    st.markdown("**Sleep Duration vs Study Hours**")
    req1 = [x for x in ["Sleep Duration", col_study_hours] if x]
    dsrc, dname = _choose_source(req1)
    if dname is None or dsrc.empty or col_study_hours is None or col_study_hours not in dsrc.columns:
        st.info("Missing columns for this chart: needs 'Sleep Duration' and a mapped 'Study Hours'.")
    else:
        tmp = dsrc.copy()
        for c in ["Sleep Duration", col_study_hours]:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        color_col = "Gender" if "Gender" in tmp.columns else None
        fig_sd_sh = px.scatter(tmp, x=col_study_hours, y="Sleep Duration", color=color_col,
                               trendline="ols",
                               hover_data=[c for c in ["Age","Occupation","BMI Category","Sleep Disorder"] if c in tmp.columns])
        st.plotly_chart(fig_sd_sh, use_container_width=True)

    # 2) Sleep Quality by University Year (box)
    st.markdown("**Sleep Quality by University Year**")
    req2 = [x for x in ["Quality of Sleep", col_univ_year] if x]
    dsrc, dname = _choose_source(req2)
    if dname is None or dsrc.empty or col_univ_year is None or col_univ_year not in dsrc.columns:
        st.info("Missing columns: needs 'Quality of Sleep' and mapped 'University Year'.")
    else:
        st.plotly_chart(px.box(dsrc, x=col_univ_year, y="Quality of Sleep", points="outliers"),
                        use_container_width=True)

    # 3) Caffeine Intake vs Sleep Duration (scatter)
    st.markdown("**Caffeine Intake vs Sleep Duration**")
    req3 = [x for x in [col_caffeine, "Sleep Duration"] if x]
    dsrc, dname = _choose_source(req3)
    if dname is None or dsrc.empty or col_caffeine is None or col_caffeine not in dsrc.columns:
        st.info("Missing columns: needs mapped 'Caffeine Intake' and 'Sleep Duration'.")
    else:
        tmp = dsrc.copy()
        for c in [col_caffeine, "Sleep Duration"]:
            tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
        color_col = "Gender" if "Gender" in tmp.columns else None
        st.plotly_chart(px.scatter(tmp, x=col_caffeine, y="Sleep Duration", color=color_col, trendline="ols"),
                        use_container_width=True)

    # 4) Physical Activity and Sleep Quality — already present above
    st.caption("*Note: 'Physical Activity vs Quality of Sleep' above covers request #4.*")

    # 5) Sleep Start and End Times — Weekdays vs Weekends
    st.markdown("**Sleep Start and End Times — Weekdays vs Weekends**")
    needed = [x for x in [col_sleep_start, col_sleep_end] if x]
    dsrc_date, dname_date = _choose_source(needed + ([col_date] if col_date else []))
    dsrc_only, dname_only = _choose_source(needed)
    dsrc = dsrc_date if dname_date is not None else dsrc_only

    def _both_present(df_, cols_): return isinstance(df_, pd.DataFrame) and not df_.empty and all(c in df_.columns for c in cols_)
    if not _both_present(dsrc, needed):
        st.info("Missing columns: map 'Sleep Start' and 'Sleep End' (and optionally 'Date').")
    else:
        tmp = dsrc.copy()
        exist_time_cols = [c for c in [col_sleep_start, col_sleep_end] if c in tmp.columns]
        for c in exist_time_cols:
            tmp[c] = pd.to_datetime(tmp[c], errors="coerce")

        if col_date and col_date in tmp.columns:
            tmp[col_date] = pd.to_datetime(tmp[col_date], errors="coerce")
            dow = tmp[col_date].dt.dayofweek
            tmp["Day Type"] = np.where(dow.isin([4,5]), "Weekend", "Weekday")  # Fri/Sat weekend
        elif "Day Type" in tmp.columns:
            tmp["Day Type"] = tmp["Day Type"].astype(str)
        else:
            tmp["Day Type"] = "Unknown"

        tmp = tmp.dropna(subset=exist_time_cols).copy()
        if tmp.empty or (tmp["Day Type"] == "Unknown").all():
            st.info("Need a Date (or an existing 'Day Type') to separate weekdays vs weekends.")
        else:
            tmp["Start_m"] = tmp[col_sleep_start].dt.hour * 60 + tmp[col_sleep_start].dt.minute
            tmp["End_m"]   = tmp[col_sleep_end].dt.hour   * 60 + tmp[col_sleep_end].dt.minute
            if col_date and col_date in tmp.columns and tmp[col_date].notna().any():
                plot_df = tmp.dropna(subset=[col_date]).copy()
                plot_df = plot_df.melt(id_vars=[col_date, "Day Type"], value_vars=["Start_m", "End_m"],
                                       var_name="Metric", value_name="Minutes")
                plot_df["Metric"] = plot_df["Metric"].map({"Start_m": "Sleep Start", "End_m": "Sleep End"})
                fig_time = px.line(plot_df.sort_values(col_date), x=col_date, y="Minutes",
                                   color="Metric", line_dash="Day Type", hover_data=["Day Type"])
                fig_time.update_layout(yaxis_title="Time (minutes since midnight)")
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                agg = tmp.groupby("Day Type", as_index=False)[["Start_m", "End_m"]].mean(numeric_only=True)
                agg = agg.melt(id_vars=["Day Type"], value_vars=["Start_m", "End_m"],
                               var_name="Metric", value_name="Minutes")
                agg["Metric"] = agg["Metric"].map({"Start_m": "Sleep Start", "End_m": "Sleep End"})
                fig_time2 = px.line(agg, x="Day Type", y="Minutes", color="Metric")
                fig_time2.update_layout(yaxis_title="Time (minutes since midnight)")
                st.plotly_chart(fig_time2, use_container_width=True)

# ================== DATA TABLE ==================
with tab_table:
    st.subheader("Filtered Data (Primary)")
    st.dataframe(fdf, use_container_width=True)
    st.download_button("Download filtered CSV",
                       fdf.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_sleep_data.csv",
                       mime="text/csv")

# ================== SECOND DATASET (Preview only) ==================
with tab_second:
    st.subheader("Second Dataset — Built-in Preview")
    if df2.empty:
        st.error(f"Could not load '{SECOND_CSV}'. Put it next to app.py.")
    else:
        c1, c2 = st.columns([2,1])
        with c1:
            st.markdown("**Preview (first 200 rows)**")
            st.dataframe(df2.head(200), use_container_width=True)
        with c2:
            st.markdown("**Shape**"); st.write(df2.shape)
            st.markdown("**Columns & dtypes**")
            st.write(pd.DataFrame({'column': df2.columns, 'dtype': [str(t) for t in df2.dtypes]}))
        st.download_button("Download second dataset (as-is)",
                           df2.to_csv(index=False).encode("utf-8"),
                           file_name="second_dataset.csv", mime="text/csv")

# ================== CONCLUSION ==================
with tab_end:
    st.subheader("Conclusion")
    st.write(
        "- الداتا الثانية الآن **جاهزة تلقائيًا** من نفس المجلد (student_sleep_patterns.csv).\n"
        "- كل الرسومات في نفس التبويب Visualizations.\n"
        "- إذا ودّك نحدد أعمدة مختلفة للأسماء، استخدم **Column Mapping**.\n"
        "- تقدر تفعل الدمج من السايدبار لو تحب."
    )
