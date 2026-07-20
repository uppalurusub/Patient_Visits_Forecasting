import streamlit as st, pandas as pd, plotly.express as px
def show_timeseries(result):
    pred=result.get("prediction",[])
    df=pd.DataFrame({"Day":range(1,len(pred)+1),"Forecast":pred})
    st.dataframe(df,use_container_width=True)
    st.plotly_chart(px.line(df,x="Day",y="Forecast",markers=True),use_container_width=True)
def show_regression(result,data):
    st.metric("Predicted Patient Count",round(float(result.get("prediction",0)),2))
    st.dataframe(pd.DataFrame({"Feature":list(data.keys()),"Value":list(data.values())}),use_container_width=True)
