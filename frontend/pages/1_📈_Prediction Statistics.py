import streamlit as st
import pandas as pd
import json
import requests
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
from streamlit_echarts import st_echarts


# st.title('A title with _italics_ :blue[colors] and emojis :sunglasses:')
st.set_page_config(layout="wide")
reduce_header_height_style = """
    <style>
        div.block-container {padding-top:1rem;}
    </style>
"""
st.markdown(reduce_header_height_style, unsafe_allow_html=True)
hide_default_format = """
       <style>
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)

st.title("Prediction statistics on the new data ( from 01-01-2023 upto current day )")

with st.sidebar:
    st.warning(
            "The model was trained on the data across 10 years from 01-12-2011 to 31-12-2022. \
             The prediction statistics are derived from the new data since 01-01-2023 upto now. The trained model works quite well given \
             that the complaints were quite healthy.",
            icon="📌",
        )

# Statitistics on how our model is working on the new data which is updated daily

df_inf = pd.read_parquet("inferences/inferences.parquet")
labels = list(df_inf["Product"].unique())

perc_correctly_classified = {}
for label in labels:
    df_subset = df_inf[df_inf["Product"] == label]
    print(label)
    perc_correctly_classified[label] = np.round(
        (df_subset[df_subset["Product"] == df_subset["Product_pred"]].shape[0])
        / df_subset.shape[0]
        * 100,
        2,
    )


df_pred = (
    pd.DataFrame([perc_correctly_classified])
    .T.reset_index()
    .rename(columns={"index": "label", 0: "% correctly classified"})
)


col1, col2 = st.columns(2)

with col2:
    st.write("**% of correctly classified complaints based on the complaint type**")

    options = {
        "tooltip": {"trigger": "item"},
        "legend": {"top": "5%", "left": "center"},
        "series": [
            {
                "name": "Complaint type",
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#fff",
                    "borderWidth": 2,
                },
                "label": {"show": False, "position": "center"},
                "emphasis": {
                    "label": {"show": True, "fontSize": "15", "fontWeight": "bold"}
                },
                "labelLine": {"show": False},
                "data": [
                    {"value": value, "name": key}
                    for key, value in perc_correctly_classified.items()
                ],
            }
        ],
    }
    st_echarts(options=options, height="550px")


# with col2:
#    st.dataframe(
#        df_pred,
#        use_container_width=True,
#        height=400,
#        hide_index=True,
#    )

with col1:
    #    st.markdown("**Total number of new complaints :** :red[{}]".format(df_inf.shape[0]))
    st.write("**% of correctly classified complaints**")

    perc_value = (
        df_inf[df_inf["Product"] == df_inf["Product_pred"]].shape[0]
        / df_inf.shape[0]
        * 100.0
    )

    option = {
        "tooltip": {"formatter": "{a} <br/>{b} : {c}%"},
        "series": [
            {
                "name": "%",
                "type": "gauge",
                "startAngle": 180,
                "endAngle": 0,
                "progress": {"show": "true"},
                "radius": "100%",
                "itemStyle": {
                    "color": "#58D9F9",
                    "shadowColor": "rgba(0,138,255,0.45)",
                    "shadowBlur": 10,
                    "shadowOffsetX": 2,
                    "shadowOffsetY": 2,
                    "radius": "55%",
                },
                "progress": {"show": "true", "roundCap": "true", "width": 15},
                "pointer": {"length": "60%", "width": 8, "offsetCenter": [0, "5%"]},
                "detail": {
                    "valueAnimation": "true",
                    "formatter": "{value}%",
                    "backgroundColor": "#58D9F9",
                    "borderColor": "#999",
                    "borderWidth": 4,
                    "width": "60%",
                    "lineHeight": 20,
                    "height": 20,
                    "borderRadius": 188,
                    "offsetCenter": [0, "40%"],
                    "valueAnimation": "true",
                },
                "data": [{"value": np.round(perc_value, 2), "name": "%"}],
            }
        ],
    }

    st_echarts(options=option, key="1")

    st.markdown("**Total number of new complaints : :red[{}]**".format(df_inf.shape[0]))
    st.markdown(
        "**Total number of new complaints correctly classified by the model :  :green[{}]**".format(
            df_inf[df_inf["Product"] == df_inf["Product_pred"]].shape[0]
        )
    )


st.write(":heavy_minus_sign:" * 60)


col3, col4 = st.columns(2)

with col3:
    st.markdown('**Points to note :**')
    st.write(
        """
             1. :red[Mortage] and :red[Credit reporting, repair, or other] have the highest correctly classified proprotion 
             2. :red[Money transfer, virtual currency, or money service] has the lowest correctly classified proprotion
             """
    )

with col4:
    st.dataframe(
        df_pred,
        use_container_width=True,
        hide_index=True,
    )
