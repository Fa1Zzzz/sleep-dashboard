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

        # ✅ ✅ ✅  ***تشارت 4 محذووووف بالكامل من هنا***

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
            st.info("وفر أعمدة Weekday/Weekend أو وفّر أعمدة موحّدة 'Sleep Start' و 'Sleep End'.")
