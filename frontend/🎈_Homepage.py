import streamlit as st
import pandas as pd
import json
import requests
import time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

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
       #MainMenu {visibility: hidden; }
       footer {visibility: hidden;}
       </style>
       """
st.markdown(hide_default_format, unsafe_allow_html=True)


st.title("Automated Customer Complaint Ticket Classification :computer:")


with st.sidebar:
    st.markdown("## How it works? :thought_balloon:")
    st.write(
        """
             1. Takes an input complaint from the user   
             2. Convert the text to features using Tfidf Vectorizer
             3. Get model inferences (label and probabilities) using HTTP request on [FastAPI](https://backend_con-1-k4288402.deta.app/docs)
             4. Display the predicted class of the complaint
             """
    )

    st.markdown("## Practical Applications? :ballot_box_with_ballot:")
    st.write(
        """
             1. Instantly classify a complaint for faster response time to customers
             2. Free up manpower on routine tasks to increase efficiency of an organization
             3. Organize lengthy complaints without reading them to reduce strain on customer service department
             """
    )


@st.cache_data
def load_model_artifacts():
    tokenizer = AutoTokenizer.from_pretrained("Kayvane/distilbert-complaints-product")
    model = AutoModelForSequenceClassification.from_pretrained(
        "Kayvane/distilbert-complaints-product"
    )
    return tokenizer, model


tokenizer, model = load_model_artifacts()


sample_txt = "I owe XXXX dollars for lab work I had done a few months ago. Due to my current financial situation, I cant pay it right now. I am a XXXX XXXX XXXX XXXX and do not have excess income. I can pay for food, and living expenses ( living paycheck to paycheck ). The first time I answered the phone for this company he asked me for my credit card number before I could even process who the call was from. When I told him I am not comfortable giving out information over the phone he refused to send me a paper bill ( Its been over a month and I have still not received one ). I hung up because I was unsure if this was even a real collection agency. They have started calling every day, and sometimes several times per day, leaving long messages ( some over two minutes long ) of them breathing loudly into the phone. Im very close to changing my phone number because of this situation. If I had the financial ability to pay the bill, I would have by now. Its frustrating being harassed by this company, and the messages of them breathing into the phone make me very uncomfortable."
text_from_user = st.text_area(
    "Enter the complaint below (a test sample has been shown below)", sample_txt
)

if st.button("Submit"):
    if text_from_user != "":
        inputs = {"text": text_from_user}

        ## sklearn model inferences
        response = requests.post(
            url="https://backend_con-1-k4288402.deta.app/predict", json=inputs
        )

        response = response.text
        output = json.loads(response)["complaint ticket type"]
        predict_proba = json.loads(response)["class probabilities"]
        class_labels = json.load(
            open("artifacts/data_transformation/labels_mapping.json")
        )
        class_labels = list(class_labels.values())
        df_sklearn_model = pd.DataFrame(
            list(
                zip(
                    class_labels,
                    np.round(predict_proba, 3),
                )
            ),
            columns=["Complaint type", "Probability"],
        )

        # hugging face inferences :
        inputs_tokens = tokenizer(
            text_from_user, return_tensors="pt", truncation=True, max_length=512
        )
        with torch.no_grad():
            logits = model(**inputs_tokens).logits
        predicted_class_id = logits.argmax().item()
        predicted_label = model.config.id2label[predicted_class_id]
        probabilities_scores = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
        df_hugging_face = pd.DataFrame(
            list(
                zip(
                    list(model.config.id2label.values()),
                    np.round(probabilities_scores, 3),
                )
            ),
            columns=["Complaint type", "Probability"],
        )

        with st.spinner(
            "Getting inferences from FastAPI and Hugging Face Hosted inference API..."
        ):
            time.sleep(1)

        st.warning(
            "Distillbert model predictions are more granular since it was trained on a huge data and considered all the 18 labels. For LinearSVM \
                   training, few sub-lables were combined together to form 9 major labels. Example : :red[Student loan, Vehicle loan or lease,\
                   Consumer loan, Payday loan] were combinded together as :red[Loan]",
            icon="⚠️",
        )

        col1, col2 = st.columns(2)
        with col1:
            st.write(
                "**Sklearn Model (Linear SVC) Prediction** : **:green[{}]**".format(
                    output
                )
            )

            def highlight_label(s):
                return (
                    ["background-color: lightgreen"] * len(s)
                    if s.Probability == df_sklearn_model.Probability.max()
                    else ["background-color: "] * len(s)
                )

            st.dataframe(
                df_sklearn_model.style.format({"Probability": "{:.3f}"}).apply(
                    highlight_label, axis=1
                ),
                use_container_width=True,
                height=350,
                hide_index=True,
            )

        with col2:
            st.write(
                "**Hugging Face hosted Model (distilbert) Prediction** : **:blue[{}]**".format(
                    predicted_label
                )
            )

            def highlight_label(s):
                return (
                    ["background-color: aqua"] * len(s)
                    if s.Probability == df_hugging_face.Probability.max()
                    else ["background-color:"] * len(s)
                )

            st.dataframe(
                df_hugging_face.style.format({"Probability": "{:.3f}"}).apply(
                    highlight_label, axis=1
                ),
                use_container_width=True,
                height=670,
                hide_index=True,
            )

 #       with st.sidebar:
 #           st.warning(
 #               "Distillbert model predictions are more granular since it was trained on a huge data and considered all the 18 labels. For LinearSVM \
 #                  training, few sub-lables were combined together to form 9 major labels. Example : :red[Student loan, Vehicle loan or lease,\
 #                  Consumer loan, Payday loan] were combinded together as :red[Loan]",
 #               icon="⚠️",
 #           )

    else:
        st.warning("No text entered", icon="⚠️")
