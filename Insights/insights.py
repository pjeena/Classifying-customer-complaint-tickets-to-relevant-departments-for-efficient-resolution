import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#from nltk.tokenize import RegexpTokenizer
import streamlit.components.v1 as components
from PIL import Image
#from annotated_text import annotated_text



st.set_page_config(
    page_title="Streamlit Dashboard",
    layout="wide",
    page_icon="💹",
    initial_sidebar_state="expanded",
)
with open("Insights/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
st.title("Analysing Financial complaints received by companies")
st.write('*This is a dashboard based on the data available on [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/#download-the-data) \
         which sends the complaints to the companies for response. The database is updated on a daily basis*')

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

#################################################### Calculations #############################################################################
@st.cache_data(experimental_allow_widgets=True)
def read_data():
    df = pd.read_parquet("data/raw/EDA_data.parquet")
    return df

df = read_data()
df = df.replace(
    {
        "Product": {
            "Credit reporting, credit repair services, or other personal consumer reports": "Credit reporting, repair, or other",
            "Credit reporting": "Credit reporting, repair, or other",
            "Credit card": "Credit card or prepaid card",
            "Prepaid card": "Credit card or prepaid card",
            "Student loan": "Loan",
            "Vehicle loan or lease": "Loan",
            "Payday loan, title loan, or personal loan": "Loan",
            "Consumer Loan": "Loan",
            "Payday loan": "Loan",
            "Money transfers": "Money transfer, virtual currency, or money service",
            "Virtual currency": "Money transfer, virtual currency, or money service",
        }
    },
)
df["Date received"] = pd.to_datetime(df["Date received"], format="%m/%d/%y")
#df = df.head(10000)


@st.cache_data(experimental_allow_widgets=True)
def get_details(df):
    total_num_of_complaints = df.shape[0]
    time_duration = (
        pd.to_datetime(df["Date received"]).dt.year.max()
        - pd.to_datetime(df["Date received"]).dt.year.min()
    )
    num_of_types_of_complaints = df["Product"].nunique()
    perc_timely_response = np.round(
        df[df["Timely response?"] == "Yes"].shape[0] / df.shape[0] * 100, 2
    )
    perc_consumer_disputed = np.round(
        df[df["Consumer disputed?"] == "Yes"].shape[0] / df.shape[0] * 100, 2
    )
    num_of_companies = df["Company"].nunique()

    df_product_type = (
        df["Product"]
        .value_counts()
        .reset_index()
        .rename(columns={"Product": "Complaint type", "count": "No of complaints"})
    )
    df_company = df["Company"].value_counts()[0:10].reset_index()
    df_monthly_stat = (
        pd.to_datetime(df["Date received"], format="%m/%d/%y")
        .dt.month_name()
        .value_counts()
        .reset_index()
    )
    df_daywise_stat = (
        pd.to_datetime(df["Date received"], format="%m/%d/%y")
        .dt.day_name()
        .value_counts()
        .reset_index()
    )
    df_time_plot = df.set_index("Date received").resample("MS").size().reset_index()
    df_status = df["Company response to consumer"].value_counts().reset_index()
    df_state_resolution = (
        df.groupby(["State", "Company response to consumer"])["Complaint ID"]
        .count()
        .reset_index()
    )
    df_state_resolution = df_state_resolution[
        df_state_resolution["State"] != "UNITED STATES MINOR OUTLYING ISLANDS"
    ]
    df_us_states = df["State"].value_counts().reset_index()

    return (
        total_num_of_complaints,
        time_duration,
        num_of_types_of_complaints,
        perc_timely_response,
        perc_consumer_disputed,
        num_of_companies,
        df_product_type,
        df_company,
        df_monthly_stat,
        df_daywise_stat,
        df_time_plot,
        df_status,
        df_state_resolution,
        df_us_states,
    )


(
    total_num_of_complaints,
    time_duration,
    num_of_types_of_complaints,
    perc_timely_response,
    perc_consumer_disputed,
    num_of_companies,
    df_product_type,
    df_company,
    df_monthly_stat,
    df_daywise_stat,
    df_time_plot,
    df_status,
    df_state_resolution,
    df_us_states,
) = get_details(df)

#################################################### Calculations #############################################################################


### top row

st.markdown("## ")

first_kpi, second_kpi, third_kpi = st.columns(3)


with first_kpi:
    st.image("Insights/complain.png", width=100)
    st.metric(label="**Total number of complaints**", value=total_num_of_complaints)


with second_kpi:
    st.image("Insights/calendar.png", width=100)
    st.metric(label="**Time Period of complaints in years**", value=time_duration)

with third_kpi:
    st.image("Insights/number-blocks.png", width=100)
    st.metric(
        label="**No of categories of complaints**", value=num_of_types_of_complaints
    )


### second row

st.markdown("<hr/>", unsafe_allow_html=True)

# st.markdown("## Secondary KPIs")

first_kpi, second_kpi, third_kpi = st.columns(3)


with first_kpi:
    #    st.image('Insights/complain.png',width=100)
    st.metric(
        label="**No of companies associated with the complaints**",
        value="{}".format(num_of_companies),
    )

with second_kpi:
    st.metric(
        label="**% of complaints with timely responses**",
        value="{}%".format(perc_timely_response),
    )

with third_kpi:
    st.metric(
        label="**% of complaints where the consumer disputed company’s response**",
        value="{}%".format(perc_consumer_disputed),
    )


# st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown(
    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True,
)

st.markdown(
    "## How are complaints distributed across different categories and companies ?"
)

first_chart, second_chart = st.columns(2)


with first_chart:
    fig = px.bar(
        df_product_type,
        x="Complaint type",
        y="No of complaints",
        #        color="No of complaints",
        title="Complaint category distribution",
    )
    fig.update_xaxes(tickangle=45, automargin=True)
    st.plotly_chart(fig, theme="streamlit")

with second_chart:
    fig = px.bar(
        df_company,
        x="Company",
        y="count",
        labels={"count": "No of complaints"},
        #        color="count",
        title="Companies with the most complaints",
    )
    fig.update_xaxes(tickangle=45, automargin=True)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

st.write(':paperclip: **Credit reporting, repair or other** has the highest no of complaints')
st.write(':paperclip: The largest financial firms including **Bank of America**, **Wells Fargo** and **J.P. Morgan** are near the top, which could be due simply to the size of the firms relative to others.')



st.markdown(
    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True,
)

st.markdown("## Temporal distribution of complaints")

first_chart, second_chart = st.columns(2)


with first_chart:
    fig = px.bar(
        df_monthly_stat,
        x="Date received",
        y="count",
        labels={"count": "No of complaints", "Date received": "Month"},
        title="Monthly distribution of complaints",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with second_chart:
    fig = px.bar(
        df_daywise_stat,
        x="Date received",
        y="count",
        labels={"count": "No of complaints", "Date received": "Day"},
        title="Daywise distribution of complaints",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

st.write(':paperclip: The month of **October** and the day of week **Wednesday** receives the highest no of complaints')




st.markdown(
    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True,
)

st.markdown("## Complaints by date received and distribution across US-states")

first_chart, second_chart = st.columns(2)


with first_chart:
    fig = px.line(
        df_time_plot, x="Date received", y=0, labels={"0": "No of complaints"}
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with second_chart:
    #    fig = px.bar(df_status, x='Company response to consumer', y='count',labels={'count':'No of complaints'})
    #    st.plotly_chart(fig,theme='streamlit',use_container_width=True)
    fig = go.Figure(
        data=go.Choropleth(
            locations=df_us_states["State"],  # Spatial coordinates
            z=df_us_states["count"].astype(float),  # Data to be color-coded
            locationmode="USA-states",  # set of locations match entries in `locations`
            colorscale="Reds",
            colorbar_title="No of complaints",
        )
    )

    fig.update_layout(
        title_text="Distribution of complaints across US states",
        geo_scope="usa",  # limite map scope to USA
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)



st.write(':paperclip: The no of complaints have risen steadily across the years')
st.write(':paperclip: **California** recevies the highest no of complaints which is feasible since most of the financial and tech companies are located there')



st.markdown(
    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True,
)

st.markdown("## Complaint status based on company's response")

first_chart, second_chart = st.columns(2)


with first_chart:
    fig = px.bar(
        df_status,
        x="Company response to consumer",
        y="count",
        labels={"count": "No of complaints"},
        title="Complaint status based on company response",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)

with second_chart:
    fig = px.bar(
        df_state_resolution,
        x="State",
        y="Complaint ID",
        labels={"Complaint ID": "No of complaints"},
        color="Company response to consumer",
        title="Complaint status across different US states",
    )
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)



st.write(':paperclip: Majority of the complaints have been closed by the companies with explanation and a very few complaints are in progress or pending across all US states')



### Top issues based on compalint ty

st.markdown(
    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True,
)


st.markdown("## Which issues are specifically complained about the most?")
st.write('Issues specific to a certain complaint type can be visualized below. This can vastly helpful for the companies to focus on relevant\
          topics, resolve them and provide seamless service to the customers')


option = st.selectbox("Choose a complaint type", df["Product"].unique())

col1, col2, col3 = st.columns([0.1, 10, 0.1])

with col1:
    st.write("")

with col2:
    df_top_issues = (
        df[df["Product"] == option]["Issue"].value_counts()[0:5].reset_index()
    )
    fig = px.line_polar(
        df_top_issues,
        r="count",
        theta="Issue",
        line_close=True,
        color_discrete_sequence=["red"],
        title="",
    )
    fig.update_traces(fill="toself")
    fig.update_xaxes(automargin=True)
    st.plotly_chart(fig, theme="streamlit", use_container_width=True)


with col3:
    st.write("")


st.markdown(
    """
---
Created with ❤️ by [Piyush](https://github.com/pjeena).
"""
)
