import streamlit as st
def render_sidebar():
    st.sidebar.header("Forecast Configuration")
    model=st.sidebar.selectbox("Select Model",["TIMESERIES_MODEL","REGRESSION_MODEL"])
    data={}
    if model=="TIMESERIES_MODEL":
        data["horizon"]=st.sidebar.number_input("Forecast Horizon",1,365,30)
    else:
        fields=[("patient_count",100),("admissions",20),("discharges",18),("occupancy",75.0),("year",2026),("month",7),("day",2),("dayofweek",4),("quarter",3),("weekofyear",27)]
        for k,v in fields: data[k]=st.sidebar.number_input(k.replace("_"," ").title(),value=v)
    return model,data,st.sidebar.button("Generate Prediction",use_container_width=True)
