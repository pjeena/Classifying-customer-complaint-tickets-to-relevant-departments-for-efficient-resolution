from src.components.model_trainer import read_yaml_file, model_trainer


class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = read_yaml_file()
        model_trainer(config=config)