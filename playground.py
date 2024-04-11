import boto3
import sagemaker
from sagemaker.workflow.parameters import ParameterInteger, ParameterString, ParameterFloat
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os 

# os.environ['AWS_PROFILE'] = 'Admin_R' 
boto3_session = boto3.session.Session(profile_name = 'Admin_R')

role = "arn:aws:iam::397567358266:role/Sagemaker_Role"
sess = sagemaker.Session(boto_session = boto3_session)
region = sess.boto_region_name
bucket = sess.default_bucket()

s3_client = boto3_session.client('s3')




input_raw_data_address = "./data/raw_data/womens_clothing_ecommerce_reviews.csv"
s3_raw_data_address = "raw_data/womens_clothing_ecommerce_reviews.csv"
raw_data_bucket = "raw-data-for--" + bucket


raw_data_bucket = "raw-data-for--sagemaker-ca-central-1-350203096375"

buckets_list = [buck["Name"] for buck in s3_client.list_buckets()['Buckets']]

if raw_data_bucket not in buckets_list:
    s3_client.create_bucket(Bucket = bucket, CreateBucketConfiguration={'LocationConstraint':region})

# s3_client.upload_file(input_raw_data_address, raw_data_bucket, s3_raw_data_address)

buckets = s3_client.list_buckets()['Buckets']
#%%
input_data_address = "s3://" + raw_data_bucket + "/" + s3_raw_data_address

processing_inputs = [ProcessingInput(input_name = "raw_input_data",\
                                      source = input_raw_data_address,\
                                      destination = '/opt/ml/processing/data/raw_data/')]
# s3_data_distribution_type='ShardedByS3Key')  
processing_outputs = [ProcessingOutput(output_name = "training_data",\
                                       source = '/opt/ml/processing/data/training/',\
                                       destination = "s3://" + bucket + "/data/training/",\
                                       s3_upload_mode='EndOfJob'),
                      ProcessingOutput(output_name = "validation_data",\
                                       source = '/opt/ml/processing/data/alidation_/',\
                                       destination = "s3://" + bucket + "/data/alidation_/",\
                                       s3_upload_mode='EndOfJob'),\
                      ProcessingOutput(output_name = "test_data",\
                                       source = '/opt/ml/processing/data/alidation_/',\
                                       destination = "s3://" + bucket + "/data/alidation_/",\
                                       s3_upload_mode='EndOfJob')]
    
processing_instacne_type = "ml.t3.medium"

processor = SKLearnProcessor(framework_version = "0.23-1",role = role,\
                             instance_type = 'local',\
                             base_job_name = "test_job1",
                             instance_count =1,
                             sagemaker_session = sess)
# env={'AWS_DEFAULT_REGION': region},    
processor.run("data_preparation.py", processing_inputs, processing_outputs)