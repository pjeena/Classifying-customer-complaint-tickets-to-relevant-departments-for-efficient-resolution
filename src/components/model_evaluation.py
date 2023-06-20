import os
import glob
import pickle
import yaml
import pandas as pd
import numpy as np
import scipy.sparse
from joblib import dump, load
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


def model_evaluate(config):
    mlflow.set_experiment("automatic_ticket_classification")

    with mlflow.start_run():
        mlflow.set_tag("developer", "Piyush")
        mlflow.set_tag("algorithm", "LinearSVC")

        # load model
        model = pickle.load(open(config["model_evaluation"]["model_path"], 'rb'))


        # load test data
        X_test = scipy.sparse.load_npz(config["model_trainer"]["X_test_path"])
        y_test = (
            pd.read_parquet(config["model_trainer"]["y_test_path"])
            .reset_index(drop=True)
            .values.ravel()
        )

        # get predictions
        y_pred = model.predict(X_test)
        signature = infer_signature(X_test, y_pred)

        # log model params, trained model, preprocessor and requirements.txt
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model_joblib",
            signature=signature,
            registered_model_name="LinearSVC-model",
        )
        mlflow.log_artifact(
            local_path=config["data_transformation"]["preprocessor_path"],
            artifact_path="tfidf_preprocessor",
        )



        # log evaluation metrics
        mlflow.log_metric("accuracy_test_data", accuracy_score(y_test, y_pred))
        mlflow.log_metric(
            "balanced_accuracy_test_data", balanced_accuracy_score(y_test, y_pred)
        )

        # get classification report and log it
        report = classification_report(y_test, y_pred, output_dict=True)
        df_report = pd.DataFrame(report).transpose()
        df_report.to_csv(config["model_evaluation"]["metric_file_name"])
        mlflow.log_artifact(
            local_path=config["model_evaluation"]["metric_file_name"],
            artifact_path="classification_report",
        )

        # get confusion matrix and log it
        conf_matrix = confusion_matrix(y_test, y_pred, labels=model.classes_)
        df_conf_matrix = pd.DataFrame(conf_matrix)
        df_conf_matrix.to_csv(config["model_evaluation"]["confusion_matrix_file_name"])
        mlflow.log_artifact(
            local_path=config["model_evaluation"]["confusion_matrix_file_name"],
            artifact_path="confusion_matrix",
        )


if __name__ == "__main__":
    import time

    start = time.time()

    config = read_yaml_file()
    model_evaluate(config=config)

    end = time.time()
    print(end - start)
