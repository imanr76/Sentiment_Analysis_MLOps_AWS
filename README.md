# Development of a ML NLP Project on AWS Cloud with on Cloud Deployment

## 1. Project Description
This project focuses on the development of a NLP model for sentiment analysis task using AWS Cloud. The developed model uses LSTMs and RNNs architecture. Instead of using the local machine to preprocess the data or train the model, Sagemaker jobs are used. Additionally the model is deployed to a sagemaker endpoint after the model is trained.

The following sagemaker jobs are used:
 - Processing Job: Preprocesses the data, creates a vocabulary from training corpus, divides the data into train, validation and test sets and saves them on S3. 

 - Training Job: Reads the train, validation and test datasets from S3 and trains the model. Saves the model artifacts on S3 including the model object, the training and validation losses and accuracies for each epoch and a classification report based on the test set data. 

 - Deployment Job: Deploys the trained model to a Sagemaker endpoint. This endpoint could then be invoked by sending HTTP requests. 

The dataset used is the Amazon product reviews dataset publicly available from the following S3 location: 

"s3://dlai-practical-data-science/data/raw/womens_clothing_ecommerce_reviews.csv"

## 2. Tech Stack
 - Python
 - Pytorch
 - AWS CLI
 - AWS Python SDK
 - Sagemaker Python SDK

## 3. How to run the project: 
Before running this project. please consider the following points: 
- Install the project packages using the requirements.txt file.
- Make sure you have AWS CLI installed on your machine.
- Create and populate a .env file in the root directory of the project. You can use the .env-sample file as a guide. This file is necessary for defining some environment variables which are required for running the project. 
- to run the pipeline, run the <b>main.py</b> script from within the root directory. 
<b>NOTE: you must run the main script from within the root directory, many of the scripts use relative paths which could lead to errors</b>

A list of the input parameters could be viewed in the main.py script.

## 4. Project File Struture:

- <b>main.py</b>: Runs processing, training and deployment jobs, after the endpoint becomes available, runs a sample inference from the model as well. 

- <b>inferecne.py</b>: Could be used to send HTTP requests to invoke the model endpoint and get predictions.  

- <b>processing_job.py</b>: Could be used to submit a processing job to sagemaker.

- <b>training_job.py</b>: Could be used to submit a training job to sagemaker.

- <b>deployment_job.py</b>: Could be used to deploy a trained model to sagemaker and create an endpoint for real-time inference.

- <b>src</b>: Contains low level scripts that run in sagemaker jobs containers.

- <b>.env-sample</b>: Expected environemnt variables. 

- <b>requirements.txt</b>: The requirements file containing the required packages for using the project.    

