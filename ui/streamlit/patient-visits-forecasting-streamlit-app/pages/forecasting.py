import streamlit as st
from components.sidebar import render_sidebar
from components.request_form import build_timeseries_payload,build_regression_payload
from components.response_view import show_timeseries,show_regression
from api.forecasting_api import predict_timeseries,predict_regression
def forecasting_page():
    st.title("📈 Patient Volume Forecasting")
    model,data,run=render_sidebar()
    if not run:
        st.info("Select a model and click Generate Prediction."); return
    if model=="TIMESERIES_MODEL":
        show_timeseries(predict_timeseries(build_timeseries_payload(data)))
    else:
        show_regression(predict_regression(build_regression_payload(data)),data)
