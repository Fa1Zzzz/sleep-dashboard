# app.py — Primary + Bundled Second Dataset (charts only in Visualizations)
# -----------------------------------------------------------------------
# - Loads primary dataset as before
# - Loads second dataset automatically from "student_sleep_patterns.csv"
# - Shows the 5 requested charts ONLY under the "Visualizations" tab
# - Second dataset tab = preview only (no charts)
# -----------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

# ------------------ Page Setup ------------------
st.set_page_config(page_title="Sleep Health & Lifestyle Dashboard",
                   page_icon="😴", layout="wide")

st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Primary dataset + bundled second dataset. Requested charts live in the Visualizations tab.")

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

    # ---- نسخ القيم لأسماء قياسية يعتمد عليها الرسم ----
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

    # تحويلات رقمية
    num_cols = [
        "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
        "Study Hours", "Caffeine Intake"
    ]
    for c in num_cols:
        if c in df2.columns:
            df2[c] = pd.to_numeric(df2[c], errors="coerce")

    # أنماط
    for c in ["Gender", "University Year"]:
        if c in df2.columns:
            df2[c] = df2[c].astype("string")

    # ---- تحضير أوقات Weekday/Weekend (ساعات عشرية -> دقائق) إن توفرت ----
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

# ================== VISUALIZATIONS (ALL CHARTS LIVE HERE) ==================
with tab_viz:
    st.subheader("Core Visualizations (Primary Dataset)")
    # Primary visuals
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

    st.markdown("---")
    st.subheader("Requested Quick Charts (Second Dataset)")

    def _has(df, cols):
        return isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in cols)

    if second_df.empty:
        st.warning("⚠️ لم يتم العثور على 'student_sleep_patterns.csv' بجانب app.py.")
    else:
        # 1) Sleep Duration vs Study Hours
        st.markdown("**1) Sleep Duration vs Study Hours**")
        if _has(second_df, ["Sleep Duration", "Study Hours"]):
            tmp = second_df.copy()
            color_col = "Gender" if "Gender" in tmp.columns else None
            fig_sd = px.scatter(tmp, x="Study Hours", y="Sleep Duration", color=color_col, trendline="ols",
                                hover_data=[c for c in ["Age","University Year"] if c in tmp.columns])
            st.plotly_chart(fig_sd, use_container_width=True)
        else:
            st.info("يحتاج أعمدة: 'Sleep Duration' و 'Study Hours'.")

        # 2) Sleep Quality by University Year (box)
        st.markdown("**2) Sleep Quality by University Year**")
        if _has(second_df, ["Quality of Sleep", "University Year"]):
            tmp = second_df.copy()
            fig_q = px.box(tmp, x="University Year", y="Quality of Sleep", points="outliers")
            st.plotly_chart(fig_q, use_container_width=True)
        else:
            st.info("يحتاج أعمدة: 'Quality of Sleep' و 'University Year'.")

        # 3) Caffeine Intake vs Sleep Duration
        st.markdown("**3) Caffeine Intake vs Sleep Duration**")
        if _has(second_df, ["Caffeine Intake", "Sleep Duration"]):
            tmp = second_df.copy()
            color_col = "Gender" if "Gender" in tmp.columns else None
            fig_c = px.scatter(tmp, x="Caffeine Intake", y="Sleep Duration", color=color_col, trendline="ols")
            st.plotly_chart(fig_c, use_container_width=True)
        else:
            st.info("يحتاج أعمدة: 'Caffeine Intake' و 'Sleep Duration'.")

        # (حُذِف) 4) Physical Activity vs Sleep Quality  — بناءً على طلبك

        # 5) Sleep Start and End Times — Weekdays vs Weekends
        st.markdown("**5) Sleep Start and End Times — Weekdays vs Weekends**")
        wk_set = {"Weekday_Sleep_Start","Weekend_Sleep_Start","Weekday_Sleep_End","Weekend_Sleep_End"}
        if wk_set.issubset(set(second_df.columns)) and bool(second_df["_Agg_Time_ready"].iloc[0]):
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
            fig_time2 = px.line(agg, x="Day Type", y="Minutes", color="Metric")
            fig_time2.update_layout(yaxis_title="Time (minutes since midnight)")
            st.plotly_chart(fig_time2, use_container_width=True)
        else:
            st.info("وفر أعمدة Weekday/Weekend أو وفّر أعمدة موحّدة 'Sleep Start' و 'Sleep End' (وأفضل وجود 'Date').")

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
    st.subheader("Second Dataset — Bundled Preview (No Charts Here)")
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
        "- The five requested charts are shown only under the Visualizations tab (with #4 from the second dataset removed as requested).\n"
        "- The second dataset is loaded automatically from student_sleep_patterns.csv.\n"
        "- The Second Dataset tab is for preview/inspection only."
    )
