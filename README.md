# Automatic complaint tickets classification using text classification

******************************************************************************************************************************************
Prediction dashboard link: https://classifying-customer-complaint.streamlit.app/ \
Data Visualization dashboard link: https://analysis-of-financial-complaints.streamlit.app/
******************************************************************************************************************************************


Ticket classification plays a crucial role in managing and prioritizing customer support requests, helpdesk tickets, or any form of incoming messages. Ticket logs contain valuable insights into the customer of a particular enterprise, like:

* Why they're leaving you
* Why they want a refund
* Why they can't check out their basket
* What features they want

#### Why is ticket classification important for a company?

 * Ticket tags are customer insights that can drive improvements across the business
 * Ticket tags can be used to power process automation
 * It can help freeing up manpower on routine tasks to increase the efficiency of an organization
 * Scalable

#### What's wrong with manual ticket tagging?

* Inconsistency: Large taxonomies make it difficult to consistently apply tags. They must choose from a large library of ‘reasons’ in under 10 seconds and usually, just make a snap judgment.
* Generic tags: Most conversations cover many topics and contain frustrations that could be useful to know about. However, agents will typically apply only the obvious tags, like ‘refund’, and not the various complaints that customers had.

This project automates the process of classifying tickets/complaints based on the text data provided. By categorizing tickets into predefined classes or categories, it becomes easier to assign appropriate resources and address them effectively. 


## Data Source

The data is ingested from [Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/search/?date_received_max=2023-06-19&date_received_min=2011-12-01&page=1&searchField=all&size=25&sort=created_date_desc&tab=List) which has data from 2011 up to today. The data is updated daily. The database contains only those complaints which are sent to companies for response. Each record has a **Complaint narrative** column which describes the descriptions of their experiences with that particular company in their own words. The individual details are censored in this field due to privacy. 

## Preprocessing and Feature engineering

Since we are classifying complaints based on the complaint description, it's a case of text classification. So the text corpus needs to be converted to numerical feature vectors. There are several classical as well as SOTA techniques available nowadays to do this task. TfiDF was the preferred choice since it is simple to calculate, computationally cheap, and interpretable. It enables us to gives us a way to associate each word in a document with a number that represents how relevant each word is in that document.

![image](https://miro.medium.com/v2/resize:fit:4800/format:webp/1*V9ac4hLVyms79jl65Ym_Bw.jpeg)

Using tfidf, all the complaints were transformed to feature vectors of length around 37000 after removing stopwords and punctuations. The features seemed to be quite large since the complaints were quite lengthy and the data had around 1.1 million rows.

## Model Building

After getting the data in the correct format, we feed the data to different classical ML algorithms. Eventually, Linear SVC turned about to be quite well both in terms of balanced accuracy and training time. The linear kernel is good when there are a lot of features and mapping the data to a higher dimensional space does not really improve the performance. In text classification, both the numbers of instances (document) and features (words) are quite large as can be seen above. Moreover, the no of parameters for a Linear SVC are few such as C(regularization parameter) and class_weight which can be useful for imbalanced classes. In order to compare our models accuracy, we also use a pretrained model [distilbert-complaints-product](https://huggingface.co/Kayvane/distilbert-complaints-product) from Hugging face. 

## Model evaluation

The model was evaluated on the test data and the considered metrics were precision, recall, f1-score, and confusion matrix.

#### Confusion Matrix
![image](https://github.com/pjeena/Classifying-customer-complaint-tickets-to-relevant-departments-for-efficient-resolution/blob/main/artifacts/model_evaluation/confusion_matrix.png) 


#### Classification Metrics
![image](https://github.com/pjeena/Classifying-customer-complaint-tickets-to-relevant-departments-for-efficient-resolution/blob/main/artifacts/model_evaluation/metrics.png)


## Architecture 

![image](https://github.com/pjeena/Classifying-customer-complaint-tickets-to-relevant-departments-for-efficient-resolution/blob/main/resources/MLpipeline.jpeg)

A robust Machine learning pipeline is built such that the process can be automated. Orchestration tools **Prefect** is used to maintain a workflow of tasks in a pipeline to track,visualize, logging and monitoring. Similarly MLflow is used to log metrics, parameters, models and other artifacts by running experiments such that different trained models can be compared with each other and evaluate the best model for production.

#### Prefect

Prefect is an orchestrator for data-intensive workflows. It can help in scheduling, retrying, logging, caching , notifications and observability. Its quite easy to visualize the flow in the Prefect UI as shown below for the Data Ingestion component:

![image](https://github.com/pjeena/Classifying-customer-complaint-tickets-to-relevant-departments-for-efficient-resolution/blob/main/resources/prefect.png)


#### MLflow 

MLflow can be used to track experiments to record and compare parameters and results while running the model training pipeline which can later be used for visualizing. It also provides a central model store to collaboratively manage the full lifecycle of an MLflow Model, including model versioning, stage transitions, and annotations.

![image](https://github.com/pjeena/Classifying-customer-complaint-tickets-to-relevant-departments-for-efficient-resolution/blob/main/resources/mlflow_1.png)

![image](https://github.com/pjeena/Classifying-customer-complaint-tickets-to-relevant-departments-for-efficient-resolution/blob/main/resources/mlflow_2.png)

![image](https://github.com/pjeena/Classifying-customer-complaint-tickets-to-relevant-departments-for-efficient-resolution/blob/main/resources/mlflow_3.png)

All the metrics, parameters, models and metrics are logged in MLflow as shown in the above figures. Models can be put in staging, production or archived depending on the data/concept drift or chamge in business needs.


## Deployment

The model was deployed on [FastAPI](https://backend_con-1-k4288402.deta.app/docs) by dockerizing it. The FastAPI backend was connected to frontend via a user friendly Streamlit web app which can be accessed [here](https://classifying-customer-complaint.streamlit.app/). 


The pipeline is triggered automatically whenever new data is available, ensuring that the model is always up-to-date and accurate.

Note : **Github Actions is not entirely accurate to trigger the pipeline at the scheduled time. Therefore, sometimes the dashboard might not reflect the forecasts for the next 2 hours.**
   
## Installation and Usage

The ML pipeline has been structured in a very systematic way. The codebase is structured into 5 components under the src/components folder 
which are as follows :

`data_ingestion.py` , `data_validation.py`,  `data_transformation.py`,  `model_trainer.py`,   `model_evaluation.py`

The pipeline (src/pipeline) follows the same flow :

`stage_01_data_ingestion.py` , ➡ `stage_01_data_validation.py`, ➡  `stage_01_data_transformation.py`, ➡ `stage_01_model_trainer.py`,  ➡ `stage_01_model_evaluation.py`

The pipeline can be triggered by just running the main file `ML_pipeline.py`:

`python ML_pipeline.py`

It will complete all 5 steps and save the preprocessor, model, and metrics under the artifacts folder. It will also log all these in MLflow UI.

In order to get the web app to run:

`python frontend/🎈_Homepage.py` 


## Conclusion

Tagging tickets has always been an important task within customer service. However, with increasing amounts of customers taking to social media platforms, review sites, and more to complain, it has become harder for support agents to stay on top of all their incoming tickets. Here we see how ML-based techniques can be used to automate ticket tagging through organizations are able to complete simple tasks faster and more accurately and free human agents to focus on more fulfilling tasks. It's also scalable, and eliminates the possibility extra agents if there’s a sudden spike in tickets.

## Future work

This project provides a foundation for further development and improvement. Some possible areas for future work include:

**Topic summarization** to provide short abstract of each complaint

**Prioritization of complaints**: using sentiment analysis to filter urgent complaints.

