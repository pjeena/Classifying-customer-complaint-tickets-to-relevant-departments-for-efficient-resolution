import pandas as pd
import numpy as np
import urllib.request as request
import yaml
import os


def download_data(config, start_date, end_date):
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