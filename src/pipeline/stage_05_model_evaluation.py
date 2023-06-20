from src.components.model_evaluation import read_yaml_file, model_evaluate




class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = read_yaml_file()
        model_evaluate(config=config)