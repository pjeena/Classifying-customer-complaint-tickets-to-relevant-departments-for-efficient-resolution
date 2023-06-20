import pandas as pd
import numpy as np
import urllib.request as request
import yaml
import os


def read_yaml_file():
    path_to_yaml = "config.yaml"
    try:
        with open(path_to_yaml, "r") as file:
            config = yaml.safe_load(file)
            return config
    except Exception as e:
        print("Error reading the config file")


def download_file(config):
    start_date = config["data_ingestion"]["start_date"]
    end_date = config["data_ingestion"]["end_date"]
    #    dates_to_be_collected =   list(pd.date_range(start=start_date, end=end_date).astype(str))
    #    for date in dates_to_be_collected:
    try:
        filename, headers = request.urlretrieve(
            url=config["data_ingestion"]["source_URL"]
            .replace("<todate>", end_date)
            .replace("<fromdate>", start_date),
            filename=os.path.join(
                config["data_ingestion"]["local_data_file_csv"],
                "complaints_on_{}_to_{}.csv".format(start_date, end_date),
            ),
        )
    except:
        print("Error in File download")


def convert_data_into_parquet_format(config):
    start_date = config["data_ingestion"]["start_date"]
    end_date = config["data_ingestion"]["end_date"]

    df = pd.read_csv(
        os.path.join(
            config["data_ingestion"]["local_data_file_csv"],
            "complaints_on_{}_to_{}.csv".format(start_date, end_date),
        )
    )

    df.to_parquet(
        os.path.join(
            config["data_ingestion"]["local_data_file_parquet"],
            "complaints_on_{}_to_{}.parquet".format(start_date, end_date),
        )
    )


if __name__ == "__main__":
    config = read_yaml_file()
    print(config["data_ingestion"]["start_date"])
    print(config["data_ingestion"]["end_date"])
    download_file(config)
    convert_data_into_parquet_format(config)
