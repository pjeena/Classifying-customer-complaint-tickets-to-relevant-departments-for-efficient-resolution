import os
import glob
import pickle
import yaml
import pandas as pd
import numpy as np
import scipy.sparse
import json
from stop_words import get_stop_words
from spacy.lang.en.stop_words import STOP_WORDS as en_stop
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split


def read_yaml_file():
    path_to_yaml = "config.yaml"
    try:
        with open(path_to_yaml, "r") as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print("Error reading the config file")


def read_raw_data_and_select_relevant_columns(config):
    start_date = config["data_ingestion"]["start_date"]
    end_date = config["data_ingestion"]["end_date"]

    df = pd.read_parquet(
        os.path.join(
            config["data_ingestion"]["local_data_file_parquet"],
            "complaints_on_{}_to_{}.parquet".format(start_date, end_date),
        )
    )

    columns_relevant = ["Consumer complaint narrative", "Product"]

    return df[columns_relevant]


def preprocess_data(data,config):
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

    data["complaint_category_id"] = data["Product"].factorize()[0]
    dict_label_mapping = (
        data[["Product", "complaint_category_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    dict_label_mapping.to_csv(config['data_transformation']['labels_mapping'] + 'labels_mapping.csv')


    dict_label_mapping_json = dict(zip(data['complaint_category_id'], data['Product']))
    with open(config['data_transformation']['labels_mapping']+ 'labels_mapping.json', "w") as file:
        json.dump(dict_label_mapping_json, file)

    return data


def train_test_split_data(data):
    train, test = train_test_split(
        data, test_size=0.25, random_state=42, stratify=data["Product"]
    )

    return train, test


def text_to_vector_preprocessor(train,config):
    tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5,ngram_range=(1,1),stop_words='english')
    tfidf.fit(train["Consumer complaint narrative"])
    pickle.dump(tfidf, open(config["data_transformation"]["preprocessor_path"], "wb"))


def convert_data_to_features(train, test,config):
    preprocessor = pickle.load(
        open(config["data_transformation"]["preprocessor_path"], "rb")
    )
    X_train = preprocessor.transform(train["Consumer complaint narrative"])
    X_test = preprocessor.transform(test["Consumer complaint narrative"])
    y_train = train["complaint_category_id"]
    y_test = test["complaint_category_id"]

    scipy.sparse.save_npz(config["data_transformation"]["X_train_path"], X_train)
    scipy.sparse.save_npz(config["data_transformation"]["X_test_path"], X_test)
    pd.DataFrame(y_train).to_parquet(config["data_transformation"]["y_train_path"])
    pd.DataFrame(y_test).to_parquet(config["data_transformation"]["y_test_path"])


if __name__ == "__main__":
    import time
    start = time.time()

    config = read_yaml_file()
    df_complaints = read_raw_data_and_select_relevant_columns(config)
    df_complaints_preprocessed = preprocess_data(data=df_complaints,config=config)
    train_data, test_data = train_test_split_data(data=df_complaints_preprocessed)
    text_to_vector_preprocessor(train=train_data,config=config)
    convert_data_to_features(train=train_data, test=test_data,config=config)

    end = time.time()
    print(end - start)
