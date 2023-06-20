from src.components.data_validation import read_yaml_file, validate_raw_data_file_exist



class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = read_yaml_file()
        validate_raw_data_file_exist(config)
