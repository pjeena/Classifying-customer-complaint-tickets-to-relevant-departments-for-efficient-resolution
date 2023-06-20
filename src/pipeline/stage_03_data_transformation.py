from sklearn.model_selection import train_test_split
from src.components.data_transformation import (
    read_yaml_file,
    read_raw_data_and_select_relevant_columns,
    preprocess_data,
    train_test_split_data,
    text_to_vector_preprocessor,
    convert_data_to_features,
)


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = read_yaml_file()
        df_complaints = read_raw_data_and_select_relevant_columns(config)
        df_complaints_preprocessed = preprocess_data(data=df_complaints,config=config)
        train_data, test_data = train_test_split_data(df_complaints_preprocessed)
        text_to_vector_preprocessor(train=train_data,config=config)
        convert_data_to_features(train=train_data, test=test_data,config=config)
