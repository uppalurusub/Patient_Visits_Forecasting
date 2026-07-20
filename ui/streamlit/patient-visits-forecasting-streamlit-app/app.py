import streamlit as st
from pages.forecasting import forecasting_page
st.set_page_config(page_title="Patient Volume Forecasting",page_icon="📈",layout="wide")
forecasting_page()
