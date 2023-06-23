import os
import glob
import pickle
import yaml
import pandas as pd
import numpy as np
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


def download_data(config, start_date, end_date):
    try:
        filename, headers = request.urlretrieve(
            url=config["model_inferences"]["source_URL"]
            .replace("<todate>", end_date)
            .replace("<fromdate>", start_date),
            filename=os.path.join(
                config["model_inferences"]["data_path"],
                "complaints_on_{}_to_{}.csv".format(start_date, end_date),
            ),
        )
    except:
        print("Error in File download")


def convert_data_into_parquet_format(config, start_date, end_date):
    df = pd.read_csv(
        os.path.join(
            config["model_inferences"]["data_path"],
            "complaints_on_{}_to_{}.csv".format(start_date, end_date),
        )
    )

    df.to_parquet(
        os.path.join(
            config["model_inferences"]["data_path"],
            "complaints_on_{}_to_{}.parquet".format(start_date, end_date),
        )
    )


def read_raw_data_and_select_relevant_columns(config, start_date, end_date):
    df = pd.read_parquet(
        os.path.join(
            config["model_inferences"]["data_path"],
            "complaints_on_{}_to_{}.parquet".format(start_date, end_date),
        )
    )

    columns_relevant = ["Consumer complaint narrative", "Product"]

    return df[columns_relevant]


def preprocess_data(data, config):
    data = data.dropna().reset_index(drop=True)
    data = data.drop_duplicates(subset="Consumer complaint narrative").reset_index(
        drop=True
    )

    data = data.replace(
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

    data.to_parquet(
        os.path.join(
            config["model_inferences"]["data_path"],
            "complaints_preprocessed.parquet",
        )
    )


def model_predictions():
    preprocessor = pickle.load(
        open(config["data_transformation"]["preprocessor_path"], "rb")
    )

    model = pickle.load(open(config["model_trainer"]["model_path"], "rb"))

    df = pd.read_parquet(
        os.path.join(
            config["model_inferences"]["data_path"],
            "complaints_preprocessed.parquet",
        )
    )


    X_test = df['Consumer complaint narrative']
    X_test = preprocessor.transform(df['Consumer complaint narrative'])

    y_pred = model.predict(X_test)
    y_pred = list(map(str, y_pred))

    dict_labels = json.load(
            open(config['data_transformation']['labels_mapping']+ 'labels_mapping.json')
        )
    y_pred_label = list(map(dict_labels.get, y_pred))
    df['Product_pred'] = y_pred_label
    df = df[['Product', 'Product_pred']]
    df.to_parquet(config['model_inferences']['predictions_path'] + 'inferences.parquet')
    print(preprocessor, model)


if __name__ == "__main__":
    config = read_yaml_file()
    start_date = "2023-01-01"
    end_date = date.today().strftime("%Y-%m-%d")
    print(start_date, end_date)
    download_data(config, start_date, end_date)
    convert_data_into_parquet_format(config, start_date, end_date)
    data = read_raw_data_and_select_relevant_columns(config, start_date, end_date)
    preprocess_data(data, config)
    model_predictions()
