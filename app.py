# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ---------- 1) إعداد الصفحة ----------
st.set_page_config(page_title="Sleep Health & Lifestyle Dashboard",
                   page_icon="😴",
                   layout="wide")

st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Interactive dashboard to explore sleep, lifestyle, and health factors.")

# ---------- 2) تحميل وتنظيف البيانات ----------
@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)

    # تأكيد الأنواع (الأرقام أرقام – النصوص نصوص)
    num_cols = ["Age", "Sleep Duration", "Quality of Sleep",
                "Physical Activity Level", "Stress Level", "Heart Rate", "Daily Steps"]
    for c in num_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    cat_cols = ["Gender", "Occupation", "BMI Category", "Sleep Disorder"]
    for c in cat_cols:
        if c in df.columns:
            df[c] = df[c].astype("string")

    # توحيد فئات BMI (لو الملف فيه "Normal Weight")
    if "BMI Category" in df.columns:
        df["BMI Category"] = (df["BMI Category"]
                              .str.replace("Normal Weight", "Normal", case=False)
                              .str.title())

    # تفكيك ضغط الدم إلى انقباضي/انبساطي (اختياري)
    if "Blood Pressure" in df.columns:
        bp = df["Blood Pressure"].str.extract(r"(?P<Systolic>\d+)\s*/\s*(?P<Diastolic>\d+)")
        df[["Systolic", "Diastolic"]] = bp.astype("float")

    # متغيرات مشتقة
    if "Sleep Duration" in df.columns:
        df["Short Sleep (<6h)"] = (df["Sleep Duration"] < 6).map({True: "Yes", False: "No"})
    return df

df = load_data("Sleep_health_and_lifestyle_dataset.csv")

# حماية لو الملف ناقص
required = ["Age","Gender","Occupation","Sleep Duration","Quality of Sleep","Physical Activity Level",
            "Stress Level","BMI Category","Sleep Disorder","Heart Rate"]
missing = [c for c in required if c not in df.columns]
if missing:
    st.error(f"الأعمدة التالية مفقودة من الملف: {missing}")
    st.stop()

# ---------- 3) فلاتر جانبية ----------
st.sidebar.header("Filters")

age_min, age_max = int(df["Age"].min()), int(df["Age"].max())
age_range = st.sidebar.slider("Age Range", min_value=age_min, max_value=age_max,
                              value=(age_min, age_max), step=1)

gender_sel = st.sidebar.multiselect("Gender", options=sorted(df["Gender"].dropna().unique().tolist()),
                                    default=sorted(df["Gender"].dropna().unique().tolist()))
occ_sel = st.sidebar.multiselect("Occupation", options=sorted(df["Occupation"].dropna().unique().tolist()),
                                 default=sorted(df["Occupation"].dropna().unique().tolist()))
bmi_sel = st.sidebar.multiselect("BMI Category", options=sorted(df["BMI Category"].dropna().unique().tolist()),
                                 default=sorted(df["BMI Category"].dropna().unique().tolist()))
disorder_sel = st.sidebar.multiselect("Sleep Disorder", options=sorted(df["Sleep Disorder"].dropna().unique().tolist()),
                                      default=sorted(df["Sleep Disorder"].dropna().unique().tolist()))

# تطبيق الفلاتر
fdf = df[
    df["Age"].between(age_range[0], age_range[1]) &
    df["Gender"].isin(gender_sel) &
    df["Occupation"].isin(occ_sel) &
    df["BMI Category"].isin(bmi_sel) &
    df["Sleep Disorder"].isin(disorder_sel)
].copy()

st.sidebar.metric("Rows after filter", len(fdf))

# ---------- 4) KPIs ----------
st.subheader("Overview (KPIs)")
k1, k2, k3, k4, k5, k6 = st.columns(6)
k1.metric("Avg Sleep (h)", f"{fdf['Sleep Duration'].mean():.2f}")
k2.metric("Avg Quality (0-10)", f"{fdf['Quality of Sleep'].mean():.2f}")
k3.metric("Avg Stress (0-10)", f"{fdf['Stress Level'].mean():.2f}")
k4.metric("Avg Activity", f"{fdf['Physical Activity Level'].mean():.2f}")
k5.metric("Avg Heart Rate", f"{fdf['Heart Rate'].mean():.1f}")
k6.metric("Sleep Disorders %", f"{(fdf['Sleep Disorder'].ne('None').mean()*100):.1f}%")

# ---------- 5) تبويبات ----------
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(
    ["Sleep Analysis", "Lifestyle & Health", "Disorders", "Demographics", "Correlations", "Data"]
)

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Sleep Duration Distribution**")
        st.plotly_chart(px.histogram(fdf, x="Sleep Duration", nbins=20, marginal="box",
                                     opacity=0.9, template="plotly"), use_container_width=True)
    with c2:
        st.markdown("**Sleep Duration vs Quality of Sleep**")
        st.plotly_chart(px.scatter(fdf, x="Sleep Duration", y="Quality of Sleep",
                                   color="Gender", hover_data=["Age","Occupation","BMI Category","Sleep Disorder"],
                                   trendline="ols"),
                        use_container_width=True)
    st.markdown("**Age vs Sleep Duration**")
    st.plotly_chart(px.scatter(fdf, x="Age", y="Sleep Duration", color="Sleep Disorder",
                               hover_data=["Gender","BMI Category","Occupation"], template="plotly"),
                    use_container_width=True)

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Physical Activity vs Quality**")
        st.plotly_chart(px.scatter(fdf, x="Physical Activity Level", y="Quality of Sleep",
                                   color="Gender", hover_data=["Age","BMI Category"], trendline="ols"),
                        use_container_width=True)
    with c2:
        st.markdown("**Stress vs Sleep Duration**")
        st.plotly_chart(px.scatter(fdf, x="Stress Level", y="Sleep Duration",
                                   color="Sleep Disorder", hover_data=["Age","Gender","BMI Category"],
                                   trendline="ols"),
                        use_container_width=True)

    st.markdown("**Heart Rate Distribution**")
    st.plotly_chart(px.histogram(fdf, x="Heart Rate", nbins=25, template="plotly"),
                    use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.markdown("**BMI Category**")
        st.plotly_chart(px.pie(fdf, names="BMI Category", hole=0.35), use_container_width=True)
    with c4:
        if {"Systolic","Diastolic"}.issubset(fdf.columns):
            st.markdown("**Blood Pressure (Boxplot)**")
            bp_melt = fdf.melt(value_vars=["Systolic","Diastolic"], var_name="Type", value_name="mmHg")
            st.plotly_chart(px.box(bp_melt, x="Type", y="mmHg", color="Type"), use_container_width=True)

with tab3:
    st.markdown("**Sleep Disorder Breakdown**")
    st.plotly_chart(px.bar(fdf["Sleep Disorder"].value_counts().rename_axis("Disorder").reset_index(name="Count"),
                           x="Disorder", y="Count", text="Count"), use_container_width=True)

    st.markdown("**Comparisons (Disorder vs None)**")
    comp_cols = ["Sleep Duration","Quality of Sleep","Stress Level","Physical Activity Level","Heart Rate"]
    m = fdf.assign(HasDisorder=fdf["Sleep Disorder"].ne("None").map({True:"Has Disorder", False:"No Disorder"}))
    means = m.groupby("HasDisorder")[comp_cols].mean().round(2).reset_index().melt(
        id_vars="HasDisorder", var_name="Metric", value_name="Mean"
    )
    st.plotly_chart(px.bar(means, x="Metric", y="Mean", color="HasDisorder", barmode="group",
                           text="Mean"), use_container_width=True)

with tab4:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Gender**")
        st.plotly_chart(px.pie(fdf, names="Gender", hole=0.35), use_container_width=True)
    with c2:
        st.markdown("**Age Distribution**")
        st.plotly_chart(px.histogram(fdf, x="Age", nbins=25), use_container_width=True)

    st.markdown("**Top Occupations**")
    occ_counts = (fdf["Occupation"].value_counts().head(15)
                  .rename_axis("Occupation").reset_index(name="Count"))
    st.plotly_chart(px.bar(occ_counts, x="Occupation", y="Count", text="Count"),
                    use_container_width=True)

with tab5:
    st.markdown("**Correlation Heatmap (numerical features)**")
    num = fdf[["Sleep Duration","Quality of Sleep","Physical Activity Level","Stress Level",
               "Heart Rate","Age","Systolic","Diastolic"]].select_dtypes(include=[np.number])
    if num.shape[1] >= 2 and len(fdf) > 1:
        corr = num.corr().round(2)
        st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto",
                                  color_continuous_scale="RdBu", origin="lower"),
                        use_container_width=True)
    else:
        st.info("Need more numeric columns/rows to compute correlations.")

with tab6:
    st.dataframe(fdf, use_container_width=True)
    st.download_button("Download filtered CSV", fdf.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_sleep_data.csv", mime="text/csv")
