# Automatic complaint tickets classification using text classification

Prediction dashboard link : https://classifying-customer-complaint.streamlit.app/ \
Visualization dashbaord link : https://analysis-of-financial-complaints.streamlit.app/

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

After getting the data in the correct format, we feed the data to different classical ML algorithms. Eventually, Linear SVC turned about to be quite well both in terms of balanced accuracy and training time. The linear kernel is good when there are a lot of features and mapping the data to a higher dimensional space does not really improve the performance. In text classification, both the numbers of instances (document) and features (words) are quite large as can be seen above. Moreover, the no of parameters for a Linear SVC are few such as C(regularization parameter) and class_weight which can be useful for imbalanced classes. 

## Model evaluation

The model was evaluated on the test data and the considered metrics were precision, recall, f1-score, and confusion matrix.

#### Confusion Matrix
![image](https://github.com/pjeena/Classifying-customer-complaint-tickets-to-relevant-departments-for-efficient-resolution/blob/main/artifacts/model_evaluation/confusion_matrix.png) \


#### Classification Metrics
![image](https://github.com/pjeena/Classifying-customer-complaint-tickets-to-relevant-departments-for-efficient-resolution/blob/main/artifacts/model_evaluation/metrics.png)


## CI/CD pipeline

The API is updated daily with the latest data available from the sensors. In order to get real time forecast we need to use automation so that our model can predict the forecast at the current hour and for the next 2 hours. This project uses Github Actions to do the same. 


**Data collection**: collect data from the API and insert it into BigQuery.  frequency : **hourly**

**Preprocessing**: preprocess the collected data and prepare it for machine learning. frequency : **hourly**

**Model training**: train the machine learning model on historical data. frequency : **weekly**

**Model evaluation**: evaluating the performance of the model on the data from the psat 7 days.

**Model deployment**: deploy the model on a [web-based dashboard](https://traffic-management-and-optimization-in-paris.streamlit.app/), which displays real-time traffic and historical insights on different junctions of Paris.

The pipeline is triggered automatically whenever new data is available, ensuring that the model is always up-to-date and accurate.

Note : **Github Actions is not entirely accurate to trigger the pipeline at the scheduled time. Therefore, sometimes the dashboard might not reflect the forecasts for the next 2 hours.**


## Dashboard

The inferences were visualized by projecting the traffic to google maps showing the forecasted traffic and the change in traffic from prior hour.
Link to the [dashboard](https://traffic-management-and-optimization-in-paris.streamlit.app/).


This shows the traffic in 8 major junctions of Paris. More intensity of color -> more traffic
![embed](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM/blob/main/resources/dashboard_1.jpeg)

Here, we see the forecasts from the last  7 days and upto the next 3 hours
![embed](https://github.com/pjeena/Traffic-Management-and-Optimization-using-LSTM/blob/main/resources/dashboard_2.jpeg)

The model performs quite well. One can play around with the lag values or can include more future forecast hours instead of 3. It heavily depends on the relevant use case. 
## Installation and Usage

1. Install the dependencies `pip install -r requirements.txt`

2. Go to the src folder and run the pipeline one after another :

    `python 01_data_fetching_between_dates.py` \
    `python 02_data_ingestion_into_bigquery.py` \
    `python 03_data_preprocessing.py` \
    `python 04_feature_build.py` \
    `python 05_model_trainer.py` \
    `python 06_predict_pipeline.py` 


    One might want to make a GCP account to access BigQuery otherwise step 2 can be skipped.

3. To get the dashboard, run this :

    `python 01_🎈_Main_page.py` 


## Conclusion


This project demonstrates how machine learning algorithms can be used to forecast traffic in real-time, using data from different sources. The project also shows how a CI/CD pipeline can be implemented to automate the data processing, model training, and deployment, improving the efficiency and reliability of the project. 
## Future work

This project provides a foundation for further development and improvement. Some possible areas for future work include:

**Integration with additional data sources** : incorporating data from other sources, such as social media feeds, weather or traffic cameras.

**User feedback and interaction**: gathering user feedback and incorporating it into the design and functionality of the dashboard could improve its usability and usefulness for the public and law enforcement agencies.

Overall, I really enjoyed working on this end to end project. I enjoyed the challenge of collecting and preprocessing data from multiple sources and building a machine learning model to predict crime rates in real-time. The implementation of the CI/CD pipeline was a great learning experience for me as well, and I am proud of the automation and efficiency it brought to the project.
