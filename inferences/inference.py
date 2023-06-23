import os
import glob
import pickle
import yaml
import pandas as pd
import numpy as np
import requests
import urllib.request as request
import scipy.sparse
from joblib import dump, load
import json
from datetime import date
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import mlflow
from mlflow.models.signature import infer_signature


def read_yaml_file():
    path_to_yaml = "config.yaml"
    try:
        with open(path_to_yaml, "r") as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print("Error reading the config file")


def get_data():
    url = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/?field=complaint_what_happened&sort=relevance_desc&format=json&no_aggs=false&no_highlight=false&date_received_min=2023-01-01"

    response = requests.get(url)
    data = response.json()

    results = [element['_source'] for element in data]
    df = pd.DataFrame(results)
#    df =  df[df['complaint_what_happened'] != '']
#    df = df.drop_duplicates(subset='complaint_what_happened').reset_index(drop=True)
    return df



def preprocess_data(data):
    data =  data[data['complaint_what_happened'] != '']
    data = data.drop_duplicates(subset='complaint_what_happened').reset_index(drop=True)    
    columns_relevant = ["complaint_what_happened", "product"]
    data = data[columns_relevant]
    data = data.dropna().reset_index(drop=True)
    data = data.replace(
        {
            "product": {
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

    return data



def model_predictions(df):
    preprocessor = pickle.load(
        open(config["data_transformation"]["preprocessor_path"], "rb")
    )

    model = pickle.load(open(config["model_trainer"]["model_path"], "rb"))


    X_test = df['complaint_what_happened']
    X_test = preprocessor.transform(X_test)

    y_pred = model.predict(X_test)
    y_pred = list(map(str, y_pred))

    dict_labels = json.load(
            open(config['data_transformation']['labels_mapping']+ 'labels_mapping.json')
        )
    y_pred_label = list(map(dict_labels.get, y_pred))
    df['product_pred'] = y_pred_label

    df = df[['product', 'product_pred']]
    df.to_parquet(config['model_inferences']['predictions_path'] + 'inferences.parquet')
    print(df.shape)


if __name__ == "__main__":
    config = read_yaml_file()
    df = get_data()
    df = preprocess_data(data=df)
    model_predictions(df)
