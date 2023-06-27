from src.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from src.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from src.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from src.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from src.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline
#from logging import logger


STAGE_NAME = "Data Ingestion stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   data_ingestion = DataIngestionTrainingPipeline()
   data_ingestion.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
#        logger.exception(e)
        raise e


STAGE_NAME = "Data Validation stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_validation = DataValidationTrainingPipeline()
   data_validation.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
#        logger.exception(e)
        raise e


STAGE_NAME = "Data Transformation stage"
try:
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<") 
   data_transformation = DataTransformationTrainingPipeline()
   data_transformation.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
 #       logger.exception(e)
        raise e



STAGE_NAME = "Model Trainer stage"
try: 
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_trainer = ModelTrainerTrainingPipeline()
   model_trainer.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
#        logger.exception(e)
        raise e




STAGE_NAME = "Model Evaluation stage"
try: 
   print(f">>>>>> stage {STAGE_NAME} started <<<<<<")
   model_evaluation = ModelEvaluationTrainingPipeline()
   model_evaluation.main()
   print(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
except Exception as e:
#        logger.exception(e)
        raise e
