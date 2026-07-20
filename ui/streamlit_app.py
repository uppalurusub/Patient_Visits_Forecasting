import streamlit as st
import pandas as pd
import requests
import plotly.express as px

# ==========================================================
# Configuration
# ==========================================================

TIMESERIES_API = "http://127.0.0.1:8000/api/v1/forecast/predict"
REGRESSION_API = "http://127.0.0.1:8000/api/v1/forecast/regression-predict"

st.set_page_config(
    page_title="Patient Volume Forecasting",
    page_icon="📈",
    layout="wide"
)

# ==========================================================
# Header
# ==========================================================

st.title("📈 Patient Volume Forecasting")
st.divider()

# ==========================================================
# Sidebar
# ==========================================================

st.sidebar.header("Forecast Configuration")

selected_model = st.sidebar.selectbox(
    "Select Model",
    [
        "TIMESERIES_MODEL",
        "REGRESSION_MODEL"
    ]
)

# ==========================================================
# Time Series Inputs
# ==========================================================

if selected_model == "TIMESERIES_MODEL":

    horizon = st.sidebar.number_input(
        "Forecast Horizon",
        min_value=1,
        max_value=365,
        value=30
    )

# ==========================================================
# Regression Inputs
# ==========================================================

else:

    st.sidebar.subheader("Regression Inputs")

    patient_count = st.sidebar.number_input(
        "Patient Count",
        value=100
    )

    admissions = st.sidebar.number_input(
        "Admissions",
        value=20
    )

    discharges = st.sidebar.number_input(
        "Discharges",
        value=18
    )

    occupancy = st.sidebar.number_input(
        "Occupancy",
        value=75.0
    )

    year = st.sidebar.number_input(
        "Year",
        value=2026
    )

    month = st.sidebar.number_input(
        "Month",
        min_value=1,
        max_value=12,
        value=7
    )

    day = st.sidebar.number_input(
        "Day",
        min_value=1,
        max_value=31,
        value=2
    )

    dayofweek = st.sidebar.number_input(
        "Day Of Week",
        min_value=0,
        max_value=6,
        value=4
    )

    quarter = st.sidebar.number_input(
        "Quarter",
        min_value=1,
        max_value=4,
        value=3
    )

    weekofyear = st.sidebar.number_input(
        "Week Of Year",
        min_value=1,
        max_value=53,
        value=27
    )

# ==========================================================
# Button
# ==========================================================

predict = st.sidebar.button(
    "Generate Prediction",
    use_container_width=True
)

# ==========================================================
# Prediction
# ==========================================================

if predict:

    with st.spinner("Running prediction..."):

        if selected_model == "TIMESERIES_MODEL":

            payload = {
                "model_name": "AR",
                "horizon": int(horizon)
            }

            api = TIMESERIES_API

        else:

            payload = {
                "model_name": "ELASTICNET",
                "patient_count": patient_count,
                "admissions": admissions,
                "discharges": discharges,
                "occupancy": occupancy,
                "year": year,
                "month": month,
                "day": day,
                "dayofweek": dayofweek,
                "quarter": quarter,
                "weekofyear": weekofyear
            }

            api = REGRESSION_API

        try:

            response = requests.post(
                api,
                json=payload,
                timeout=120
            )

            response.raise_for_status()

            result = response.json()

            #################################################
            # TIME SERIES
            #################################################

            if selected_model == "TIMESERIES_MODEL":

                prediction = result.get("prediction", [])

                if len(prediction) == 0:
                    st.error("No forecast returned.")
                    st.stop()

                df = pd.DataFrame({
                    "Day": range(1, len(prediction)+1),
                    "Forecast": prediction
                })

                c1, c2, c3 = st.columns(3)

                c1.metric(
                    "Model",
                    "AR"
                )

                c2.metric(
                    "Forecast Days",
                    len(df)
                )

                c3.metric(
                    "Average Forecast",
                    round(df.Forecast.mean(),2)
                )

                st.success("Forecast generated successfully.")

                st.subheader("Forecast Results")

                st.dataframe(
                    df,
                    use_container_width=True
                )

                fig = px.line(
                    df,
                    x="Day",
                    y="Forecast",
                    markers=True
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

                st.download_button(
                    "Download CSV",
                    df.to_csv(index=False),
                    file_name="forecast.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            #################################################
            # REGRESSION
            #################################################

            else:

                prediction = result.get("prediction")

                st.success("Prediction completed.")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Model",
                    "ElasticNet"
                )

                c2.metric(
                    "Predicted Patient Count",
                    round(float(prediction),2)
                )

                st.subheader("Prediction Details")

                prediction_df = pd.DataFrame(
                    {
                        "Input Feature":[
                            "Patient Count",
                            "Admissions",
                            "Discharges",
                            "Occupancy",
                            "Year",
                            "Month",
                            "Day",
                            "Day Of Week",
                            "Quarter",
                            "Week Of Year"
                        ],
                        "Value":[
                            patient_count,
                            admissions,
                            discharges,
                            occupancy,
                            year,
                            month,
                            day,
                            dayofweek,
                            quarter,
                            weekofyear
                        ]
                    }
                )

                st.dataframe(
                    prediction_df,
                    use_container_width=True
                )

                

        except requests.exceptions.ConnectionError:

            st.error("Unable to connect to FastAPI.")

        except requests.exceptions.Timeout:

            st.error("Request timed out.")

        except requests.exceptions.HTTPError:

            st.error(response.text)

        except Exception as ex:

            st.exception(ex)

else:

    st.info(
        "Select a model from the left sidebar and click Generate Prediction."
    )