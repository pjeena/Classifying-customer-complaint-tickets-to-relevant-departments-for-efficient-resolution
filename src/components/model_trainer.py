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
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, balanced_accuracy_score
import mlflow


def read_yaml_file():
    path_to_yaml = "config.yaml"
    try:
        with open(path_to_yaml, "r") as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print("Error reading the config file")


def model_trainer(config):
    # load training data
    X_train = scipy.sparse.load_npz(config["model_trainer"]["X_train_path"])
    y_train = (
        pd.read_parquet(config["model_trainer"]["y_train_path"])
        .reset_index(drop=True)
        .values.ravel()
    )

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1.0],
        "class_weight": ["balanced"],
        "dual": [True],
    }

    grid = GridSearchCV(
        LinearSVC(),
        param_grid,
        verbose=3,
        cv=2,
        scoring="balanced_accuracy",
        n_jobs=7,
        return_train_score=True,
    )

    parameters = {
        "class_weight": "balanced",
        "dual": True,
        "C": 0.1,
        "random_state": None,
    }

    model = LinearSVC(
        class_weight=parameters["class_weight"],
        dual=parameters["dual"],
        random_state=parameters["random_state"],
        C=parameters["C"],
    )
    model.fit(X_train, y_train)
    print(model.score(X_train, y_train))
    #        mlflow.log_metric("train_data_score", model.score(X_train, y_train))

    #    model = grid.best_estimator_
    #    model.fit(X_train, y_train.values.ravel())
    #    print(model.score(X_train,y_train))

    X_test = scipy.sparse.load_npz(config["model_trainer"]["X_test_path"])
    y_test = (
        pd.read_parquet(config["model_trainer"]["y_test_path"])
        .reset_index(drop=True)
        .values.ravel()
    )

    y_pred = model.predict(X_test)
    print(accuracy_score(y_test, y_pred))
    print(balanced_accuracy_score(y_test, y_pred))

    pickle.dump(model, open(config["model_trainer"]["model_path"], 'wb'))



if __name__ == "__main__":
    import time

    start = time.time()

    config = read_yaml_file()
    model_trainer(config)

    end = time.time()
    print(end - start)
