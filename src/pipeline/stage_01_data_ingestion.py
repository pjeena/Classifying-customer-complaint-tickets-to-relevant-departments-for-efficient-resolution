from src.components.data_ingestion import read_yaml_file, download_file, convert_data_into_parquet_format



class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = read_yaml_file()
#        download_file(config=config)
        convert_data_into_parquet_format(config=config)
