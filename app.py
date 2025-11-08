# app.py — Clean Version (Requested Layout)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Sleep Dashboard", page_icon="😴", layout="wide")
st.title("Sleep Health & Lifestyle Dashboard")
st.caption("Primary + Second dataset stacked in the same sections.")

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_csv(path):
    try:
        return pd.read_csv(path)
    except:
        return pd.DataFrame()

# Load both datasets
primary_df = load_csv("Sleep_health_and_lifestyle_dataset.csv")
second_df = load_csv("student_sleep_patterns.csv")

# ---------------------- CLEAN PRIMARY ----------------------
if not primary_df.empty:
    num_cols = ["Age", "Sleep Duration", "Quality of Sleep", "Physical Activity Level",
                "Stress Level", "Heart Rate", "Daily Steps"]
    for c in num_cols:
        if c in primary_df.columns:
            primary_df[c] = pd.to_numeric(primary_df[c], errors="coerce")

# ---------------------- CLEAN SECOND ----------------------
if not second_df.empty:
    rename_map = {
        "Sleep_Duration": "Sleep Duration",
        "Sleep_Quality": "Quality of Sleep",
        "Caffeine_Intake": "Caffeine Intake",
        "Study_Hours": "Study Hours",
        "Physical_Activity": "Physical Activity Level",
        "University_Year": "University Year",
    }
    second_df = second_df.rename(columns=rename_map)

# ---------------------- VISUALIZATIONS ----------------------
tab1, tab2, tab3 = st.tabs(["Visualizations", "Data Table", "Conclusion"])

with tab1:

    st.header("Requested Visualizations (Primary then Second)")

    # ================== PRIMARY ==================
    with st.expander("Primary Dataset: Sleep_health_and_lifestyle_dataset.csv", expanded=True):
        df = primary_df

        if df.empty:
            st.warning("Primary dataset missing.")
        else:
            # 1) Sleep Duration vs Study Hours
            st.subheader("1) Sleep Duration vs Study Hours — Scatter Plot")
            if "Sleep Duration" in df.columns and "Study Hours" in df.columns:
                fig = px.scatter(df, x="Study Hours", y="Sleep Duration", color="Gender")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: Sleep Duration or Study Hours")

            # 2) Sleep Quality by University Year
            st.subheader("2) Sleep Quality by University Year — Box Plot")
            if "Quality of Sleep" in df.columns and "University Year" in df.columns:
                fig = px.box(df, x="University Year", y="Quality of Sleep")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: University Year or Quality of Sleep")

            # 3) Caffeine Intake vs Sleep Duration
            st.subheader("3) Caffeine Intake vs Sleep Duration — Scatter Plot")
            if "Caffeine Intake" in df.columns and "Sleep Duration" in df.columns:
                fig = px.scatter(df, x="Caffeine Intake", y="Sleep Duration", color="Gender")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: Caffeine Intake or Sleep Duration")

            # 4) Physical Activity and Sleep Quality
            st.subheader("4) Physical Activity vs Sleep Quality — Scatter Plot")
            if "Physical Activity Level" in df.columns and "Quality of Sleep" in df.columns:
                fig = px.scatter(df, x="Physical Activity Level", y="Quality of Sleep", color="Gender")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: Physical Activity Level or Quality of Sleep")

            # 5) Sleep Start and End Times
            st.subheader("5) Sleep Start & End — Weekdays vs Weekends — Line Chart")
            if all(col in df.columns for col in ["Sleep Start", "Sleep End", "Date"]):
                tmp = df.copy()
                tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
                tmp["Sleep Start"] = pd.to_datetime(tmp["Sleep Start"], errors="coerce")
                tmp["Sleep End"] = pd.to_datetime(tmp["Sleep End"], errors="coerce")

                tmp["Day Type"] = tmp["Date"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")

                plot_df = tmp.melt(id_vars=["Date", "Day Type"],
                                   value_vars=["Sleep Start", "Sleep End"],
                                   var_name="Metric", value_name="Time")
                plot_df["Minutes"] = plot_df["Time"].dt.hour*60 + plot_df["Time"].dt.minute

                fig = px.line(plot_df, x="Date", y="Minutes", color="Metric", line_dash="Day Type")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: Sleep Start, Sleep End, or Date")

    # ================== SECOND ==================
    with st.expander("Second Dataset: student_sleep_patterns.csv", expanded=True):
        df = second_df

        if df.empty:
            st.warning("Second dataset missing.")
        else:
            # 1) Sleep Duration vs Study Hours
            st.subheader("1) Sleep Duration vs Study Hours — Scatter Plot")
            if "Sleep Duration" in df.columns and "Study Hours" in df.columns:
                fig = px.scatter(df, x="Study Hours", y="Sleep Duration", color="Gender")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: Sleep Duration or Study Hours")

            # 2) Sleep Quality by University Year
            st.subheader("2) Sleep Quality by University Year — Box Plot")
            if "Quality of Sleep" in df.columns and "University Year" in df.columns:
                fig = px.box(df, x="University Year", y="Quality of Sleep")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: University Year or Quality of Sleep")

            # 3) Caffeine Intake vs Sleep Duration
            st.subheader("3) Caffeine Intake vs Sleep Duration — Scatter Plot")
            if "Caffeine Intake" in df.columns and "Sleep Duration" in df.columns:
                fig = px.scatter(df, x="Caffeine Intake", y="Sleep Duration", color="Gender")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: Caffeine Intake or Sleep Duration")

            # 4) Physical Activity and Sleep Quality
            st.subheader("4) Physical Activity vs Sleep Quality — Scatter Plot")
            if "Physical Activity Level" in df.columns and "Quality of Sleep" in df.columns:
                fig = px.scatter(df, x="Physical Activity Level", y="Quality of Sleep", color="Gender")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: Physical Activity Level or Quality of Sleep")

            # 5) Sleep Start and End Times
            st.subheader("5) Sleep Start & End — Weekdays vs Weekends — Line Chart")
            if all(col in df.columns for col in ["Sleep Start", "Sleep End", "Date"]):
                tmp = df.copy()
                tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
                tmp["Sleep Start"] = pd.to_datetime(tmp["Sleep Start"], errors="coerce")
                tmp["Sleep End"] = pd.to_datetime(tmp["Sleep End"], errors="coerce")

                tmp["Day Type"] = tmp["Date"].dt.dayofweek.apply(lambda x: "Weekend" if x >= 5 else "Weekday")

                plot_df = tmp.melt(id_vars=["Date", "Day Type"],
                                   value_vars=["Sleep Start", "Sleep End"],
                                   var_name="Metric", value_name="Time")
                plot_df["Minutes"] = plot_df["Time"].dt.hour*60 + plot_df["Time"].dt.minute

                fig = px.line(plot_df, x="Date", y="Minutes", color="Metric", line_dash="Day Type")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Missing: Sleep Start, Sleep End, or Date")


with tab2:
    st.subheader("Primary Dataset")
    st.dataframe(primary_df, use_container_width=True)
    st.subheader("Second Dataset")
    st.dataframe(second_df, use_container_width=True)

with tab3:
    st.write("Dashboard complete.")
