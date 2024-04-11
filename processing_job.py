import boto3
import sagemaker
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput
import os
from dotenv import load_dotenv


def setup_sagemaker(local = True):
    """
    Sets up the sagemaker and boto3 sessions required for running the processing job.

    Parameters
    ----------
    local : boolean
        Whether the script is running locally or inside sagemaker notebooks.

    Returns
    -------
    role : str, obj
        ARN role for the sagemaker session.
    bucket : str
        The default bucket name for the sagemaker session.
    region : str
        Teh region of the sgaemaker and boto3 session.
    boto3_session : obj
    sagemaker_Sess : obj
    """
    # IF running the script locally
    if local:
        load_dotenv()
        role = os.getenv("ROLE")
        profile_name  = os.getenv("AWS_PROFILE")
        boto3_session = boto3.session.Session(profile_name = profile_name)
        sagemaker_Sess = sagemaker.Session(boto_session = boto3_session)
    # If running the code from a sagemaker notebook
    else:
        boto3_session = boto3.session.Session()
        sagemaker_Sess = sagemaker.Session()
        role = sagemaker_Sess.get_execution_role()
    
    region = sagemaker_Sess.boto_region_name
    bucket = sagemaker_Sess.default_bucket()
    
    return role, bucket, region, boto3_session, sagemaker_Sess

def copy_to_S3(boto3_session, bucket, region):
    """
    Copies the csv file from the original S3 location to the S3 location of the used account.
    Replaces it if it exists.

    Parameters
    ----------
    boto3_session : obj
    bucket : str
    region : str

    Returns
    -------
    None.

    """
    s3_client = boto3_session.client('s3')
    # Create the bucket if it doesn't exist yet.
    buckets_list = [buck["Name"] for buck in s3_client.list_buckets()['Buckets']]
    if bucket not in buckets_list:
        s3_client.create_bucket(Bucket = bucket, CreateBucketConfiguration = {'LocationConstraint' : region})
    # Copying the data
    s3_client.copy_object(CopySource = "dlai-practical-data-science/data/raw/womens_clothing_ecommerce_reviews.csv",\
                         Bucket = bucket, Key = "raw_data/womens_clothing_ecommerce_reviews.csv",\
                         MetadataDirective='REPLACE')
   

def run_processing_job(processing_instacne_type = "ml.t3.large", processing_instance_count = 1,\
                       max_len = 500, train_size = 0.8, validation_size = 0.15, test_size = 0.05, local = True):
    """
    Sets up the sagemaker session, prepares the data and runs a processing job using the 
    data_preparation.py script.

    Parameters
    ----------
    processing_instacne_type : str, optional
        The type of virtual machine to use. The default is "ml.t3.large".
    processing_instance_count : int, optional
        The number of instances to use for processing job. The default is 1.
    max_len : TYPE, optional
        Maximum review text sequence length. The default is 500.
    train_size : TYPE, optional
        Fraction of training data of all data. The default is 0.8.
    validation_size : TYPE, optional
        Fraction of validation data of all data. The default is 0.15.
    test_size : TYPE, optional
        Fraction of test data of all data. The default is 0.05.
    local : boolean, optional
        Whether the script is running locally or inside sagemaker notebooks.. The default is True.

    Returns
    -------
    None.

    """
    # Setting up the sagemaker session
    role, bucket, region, boto3_session, sagemaker_Sess = setup_sagemaker(local)
    # Preparing the raw data
    copy_to_S3(boto3_session, bucket, region)
    # Defining the processing job inputs
    processing_inputs = [ProcessingInput(input_name = "raw_input_data",\
                                          source = "s3://" + bucket + "/raw_data/womens_clothing_ecommerce_reviews.csv",\
                                          destination = '/opt/ml/processing/data/raw_data/')]
    # Defining the processing job outputs.
    processing_outputs = [ProcessingOutput(output_name = "training_data",\
                                           source = '/opt/ml/processing/data/training/',\
                                           destination = "s3://" + bucket + "/data/training/",\
                                           s3_upload_mode='EndOfJob'),
                          ProcessingOutput(output_name = "validation_data",\
                                           source = '/opt/ml/processing/data/validation/',\
                                           destination = "s3://" + bucket + "/data/validation/",\
                                           s3_upload_mode='EndOfJob'),\
                          ProcessingOutput(output_name = "test_data",\
                                           source = '/opt/ml/processing/data/test/',\
                                           destination = "s3://" + bucket + "/data/test/",\
                                           s3_upload_mode='EndOfJob'),
                          ProcessingOutput(output_name = "vocabulary",\
                                           source = '/opt/ml/processing/models/',\
                                           destination = "s3://" + bucket + "/models/",\
                                           s3_upload_mode='EndOfJob')]
    # Defining the processing job 
    processor = SKLearnProcessor(framework_version = "0.23-1",role = role,\
                                 instance_type = processing_instacne_type,\
                                 base_job_name = "test-job2",
                                 instance_count =1,
                                 env={'AWS_DEFAULT_REGION': region},
                                 sagemaker_session = sagemaker_Sess)
    # Running the processing job
    processor.run("data_preparation.py", processing_inputs, processing_outputs, logs=True,\
                  arguments=["--max-len", str(100),
                             "--train-size", str(train_size),
                             "--validation-size", str(validation_size),
                             "--test-size", str(test_size)])

#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    # The type of instance to run the job on
    processing_instacne_type = "ml.t3.large"
    # The number of instances to use for processing job
    processing_instance_count = 1
    # Maximum review text sequence length
    max_len = 500
    # Fraction of training data of all data
    train_size = 0.8
    # Fraction of validation data of all data
    validation_size = 0.15
    # Fraction of test data of all data
    test_size = 0.05
    
    run_processing_job(processing_instacne_type, processing_instance_count,\
                           max_len, train_size, validation_size, test_size)
    
    
    