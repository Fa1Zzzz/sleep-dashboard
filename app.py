# app.py — Primary + Bundled Second Dataset (auto charts enabled)
# -------------------------------------------------------------
# What's new vs your file:
# 1) Loads a bundled second dataset from "student_sleep_patterns.csv" (no upload UI)
# 2) Adds the 5 requested charts to Visualizations and shows them immediately
# 3) Keeps your original primary visuals/KPIs as-is
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Sleep Health & Lifestyle Dashboard",
                   page_icon="😴", layout="wide")

st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Primary dataset + bundled second dataset. Your requested charts load automatically.")

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

# ------------------ Data Load & Clean (Second - Bundled) ------------------
SECOND_PATH = "student_sleep_patterns.csv"  # ضع الملف في نفس مجلد app.py

@st.cache_data(show_spinner=False)
def load_second_bundled(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df2 = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()

    # نحاول توحيد/تحويل الأعمدة المحتملة
    # مرشحات للأسماء المحتملة (إنجليزي/اختصارات)
    def _to_num(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")

    def _to_dt(df, cols):
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_datetime(df[c], errors="coerce")

    # أشياء شائعة بالداتا الثانية
    _to_num(df2, ["Sleep Duration", "Quality of Sleep", "Physical Activity Level",
                  "Stress Level", "Heart Rate", "Daily Steps",
                  "Study Hours", "Caffeine Intake"])
    _to_dt(df2, ["Sleep Start", "Sleep End", "Date"])

    # Gender/Year لو موجودة
    if "Gender" in df2.columns:
        df2["Gender"] = df2["Gender"].astype("string")

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

# ------------------ Sidebar Filters (Primary) ------------------
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

# Apply filters (Primary)
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

# ================== VISUALIZATIONS ==================
with tab_viz:
    st.subheader("Core Visualizations (Primary Dataset)")
    # نفس رسماتك الأساسية
    fig1 = px.histogram(fdf, x="Sleep Duration", nbins=20, marginal="box", opacity=0.9)
    st.plotly_chart(fig1, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig2 = px.scatter(
            fdf, x="Sleep Duration", y="Quality of Sleep",
            color="Gender",
            hover_data=[c for c in ["Age","Occupation","BMI Category","Sleep Disorder"] if c in fdf.columns],
            trendline="ols"
        )
        st.plotly_chart(fig2, use_container_width=True)
    with c2:
        fig3 = px.scatter(
            fdf, x="Age", y="Sleep Duration",
            color="Gender",
            hover_data=[c for c in ["Occupation","BMI Category"] if c in fdf.columns]
        )
        st.plotly_chart(fig3, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        fig4 = px.scatter(
            fdf, x="Physical Activity Level", y="Quality of Sleep",
            color="Gender",
            hover_data=[c for c in ["Age","BMI Category"] if c in fdf.columns], trendline="ols"
        )
        st.plotly_chart(fig4, use_container_width=True)
    with c4:
        fig5 = px.scatter(
            fdf, x="Stress Level", y="Sleep Duration",
            color="Gender",
            hover_data=[c for c in ["Age","BMI Category"] if c in fdf.columns], trendline="ols"
        )
        st.plotly_chart(fig5, use_container_width=True)

    fig6 = px.histogram(fdf, x="Heart Rate", nbins=25)
    st.plotly_chart(fig6, use_container_width=True)

    disorder_count = (
        fdf[fdf["Sleep Disorder"] != "None"]["Sleep Disorder"]
        .value_counts()
        .rename_axis("Disorder").reset_index(name="Count")
    )
    fig_disorder = px.bar(disorder_count, x="Disorder", y="Count", text="Count")
    fig_disorder.update_traces(textposition="outside", texttemplate="%{text:.0f}")
    fig_disorder.update_layout(yaxis_title="Count", xaxis_title="Disorder", showlegend=False, height=450,
                               margin=dict(t=40, r=20, b=70, l=60))
    st.plotly_chart(fig_disorder, use_container_width=True)

    # ---------- NEW: Requested Quick Charts (Second Dataset) ----------
    st.markdown("---")
    st.subheader("Requested Quick Charts (Second Dataset)")

    def _has(df, cols):
        return isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in cols)

    def _guess(cols, *cands):
        lower = {c.lower(): c for c in cols}
        for cand in cands:
            if cand.lower() in lower:
                return lower[cand.lower()]
        for c in cols:
            cl = c.lower()
            if any(tok.lower() in cl for tok in cands):
                return c
        return None

    # إذا الثاني فاضي ننبّه مرة واحدة
    if second_df.empty:
        st.warning("⚠️ لم يتم العثور على ملف second dataset 'student_sleep_patterns.csv' في مجلد التطبيق.")
    else:
        cols_union = list(map(str, second_df.columns))

        # تخمين أسماء الأعمدة المحتملة
        col_study  = _guess(cols_union, "Study Hours", "StudyHours", "Hours of Study", "study")
        col_year   = _guess(cols_union, "University Year", "Year", "Uni Year", "Academic Year", "year")
        col_caff   = _guess(cols_union, "Caffeine Intake", "Caffeine", "Coffee Cups", "cups")
        col_start  = _guess(cols_union, "Sleep Start", "Bedtime", "SleepStart", "Start Time", "sleep start")
        col_end    = _guess(cols_union, "Sleep End", "Wakeup", "Wake Time", "SleepEnd", "End Time", "sleep end")
        col_date   = _guess(cols_union, "Date", "Day", "Record Date", "Datetime")

        # 1) Sleep Duration vs Study Hours (scatter)
        st.markdown("**1) Sleep Duration vs Study Hours**")
        if _has(second_df, ["Sleep Duration"]) and col_study:
            tmp = second_df.copy()
            for c in ["Sleep Duration", col_study]:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            color_col = "Gender" if "Gender" in tmp.columns else None
            fig_sd = px.scatter(tmp, x=col_study, y="Sleep Duration", color=color_col, trendline="ols",
                                hover_data=[c for c in ["Age","Occupation","BMI Category","Sleep Disorder"] if c in tmp.columns])
            st.plotly_chart(fig_sd, use_container_width=True)
        else:
            st.info("يحتاج أعمدة: 'Sleep Duration' و 'Study Hours' (أو ما يقابلها).")

        # 2) Sleep Quality by University Year (box)
        st.markdown("**2) Sleep Quality by University Year**")
        if _has(second_df, ["Quality of Sleep"]) and col_year:
            tmp = second_df.copy()
            fig_q = px.box(tmp, x=col_year, y="Quality of Sleep", points="outliers")
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.info("يحتاج أعمدة: 'Quality of Sleep' و 'University Year' (أو ما يقابلها).")

        # 3) Caffeine Intake vs Sleep Duration (scatter)
        st.markdown("**3) Caffeine Intake vs Sleep Duration**")
        if _has(second_df, ["Sleep Duration"]) and col_caff:
            tmp = second_df.copy()
            for c in ["Sleep Duration", col_caff]:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            color_col = "Gender" if "Gender" in tmp.columns else None
            fig_c = px.scatter(tmp, x=col_caff, y="Sleep Duration", color=color_col, trendline="ols")
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("يحتاج أعمدة: 'Sleep Duration' و 'Caffeine Intake' (أو ما يقابلها).")

        # 4) Physical Activity and Sleep Quality (scatter) — من الداتا الثانية
        st.markdown("**4) Physical Activity vs Sleep Quality**")
        if _has(second_df, ["Physical Activity Level", "Quality of Sleep"]):
            tmp = second_df.copy()
            color_col = "Gender" if "Gender" in tmp.columns else None
            fig_pa = px.scatter(tmp, x="Physical Activity Level", y="Quality of Sleep",
                                color=color_col, trendline="ols")
            st.plotly_chart(fig_pa, use_container_width=True)
        else:
            st.info("يحتاج أعمدة: 'Physical Activity Level' و 'Quality of Sleep'.")

        # 5) Sleep Start and End Times — Weekdays vs Weekends (line)
        st.markdown("**5) Sleep Start and End Times — Weekdays vs Weekends**")
        if (col_start and col_end) and (col_start in second_df.columns) and (col_end in second_df.columns):
            tmp = second_df.copy()
            tmp[col_start] = pd.to_datetime(tmp[col_start], errors="coerce")
            tmp[col_end]   = pd.to_datetime(tmp[col_end], errors="coerce")

            # Day Type
            if col_date and col_date in tmp.columns:
                tmp[col_date] = pd.to_datetime(tmp[col_date], errors="coerce")
                dow = tmp[col_date].dt.dayofweek
                tmp["Day Type"] = np.where(dow.isin([4,5]), "Weekend", "Weekday")
            elif "Day Type" in tmp.columns:
                tmp["Day Type"] = tmp["Day Type"].astype(str)
            else:
                # بدون تاريخ: نعرض متوسطات عامة فقط
                tmp["Day Type"] = "Unknown"

            tmp = tmp.dropna(subset=[col_start, col_end]).copy()

            def to_minutes(ts):
                return ts.dt.hour * 60 + ts.dt.minute

            tmp["Start_m"] = to_minutes(tmp[col_start])
            tmp["End_m"]   = to_minutes(tmp[col_end])

            if col_date and (col_date in tmp.columns) and tmp[col_date].notna().any() and (tmp["Day Type"] != "Unknown").any():
                plot_df = tmp.dropna(subset=[col_date]).copy()
                plot_df = plot_df.melt(id_vars=[col_date, "Day Type"], value_vars=["Start_m", "End_m"],
                                       var_name="Metric", value_name="Minutes")
                plot_df["Metric"] = plot_df["Metric"].map({"Start_m": "Sleep Start", "End_m": "Sleep End"})
                fig_time = px.line(plot_df.sort_values(col_date), x=col_date, y="Minutes",
                                   color="Metric", line_dash="Day Type", hover_data=["Day Type"])
                fig_time.update_layout(yaxis_title="Time (minutes since midnight)")
                st.plotly_chart(fig_time, use_container_width=True)
            else:
                # متوسطات لكل Day Type إن وجدت، وإلا Unknown
                agg = tmp.groupby("Day Type", as_index=False)[["Start_m", "End_m"]].mean(numeric_only=True)
                agg = agg.melt(id_vars=["Day Type"], value_vars=["Start_m", "End_m"],
                               var_name="Metric", value_name="Minutes")
                agg["Metric"] = agg["Metric"].map({"Start_m": "Sleep Start", "End_m": "Sleep End"})
                fig_time2 = px.line(agg, x="Day Type", y="Minutes", color="Metric")
                fig_time2.update_layout(yaxis_title="Time (minutes since midnight)")
                st.plotly_chart(fig_time2, use_container_width=True)
        else:
            st.info("يحتاج أعمدة: 'Sleep Start' و 'Sleep End' (وأفضل وجود 'Date' للفصل بين Weekday/Weekend).")

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
    st.subheader("Second Dataset — Bundled Preview")
    if second_df.empty:
        st.warning("لم يتم العثور على 'student_sleep_patterns.csv'. ضع الملف بجانب app.py ثم أعد التشغيل.")
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

# ================== CONCLUSION ==================
with tab_end:
    st.subheader("Conclusion")
    st.write(
        "- تم تفعيل الرسوم الخمسة المطلوبة وتظهر مباشرة من الداتا الثانية.\n"
        "- الداتا الثانية تُحمَّل تلقائيًا من student_sleep_patterns.csv بدون رفع.\n"
        "- إذا ما ظهر ملف الداتا الثانية، تأكد من وجوده بجانب app.py."
    )
