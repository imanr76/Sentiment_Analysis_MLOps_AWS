from dotenv import load_dotenv
import boto3
import sagemaker
from sagemaker.pytorch import PyTorchModel
import os
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer

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


def run_deployment_job(training_job_name, deployment_instance_count = 1, deployment_instance_type = "ml.t2.medium",
                       session_info = None):
    if not session_info:
        role, bucket, region, boto3_session, sagemaker_Sess = setup_sagemaker()
    else:
        role, bucket, region, boto3_session, sagemaker_Sess = session_info
    
    model_data = "s3://" + bucket +"/models/" + training_job_name + "/output/model.tar.gz"
    
    model = PyTorchModel(
        entry_point="deployment.py",
        role=role,
        model_data = model_data,
        framework_version = "1.13",
        py_version = "py39",
        sagemaker_session = sagemaker_Sess,
        source_dir = "./src",
        env = {"PYTHONUNBUFFERED" : "1"}
    )
    
    
    predictor = model.deploy(
                            initial_instance_count = deployment_instance_count,
                            instance_type = deployment_instance_type,
                            serializer = JSONSerializer(),
                            deserializer = JSONDeserializer(),
                            logs = True
                            )
    return predictor.endpoint_name


#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    deployment_instance_count = 1
    
    deployment_instance_type = "ml.t2.medium"
    
    training_job_name = 'pytorch-training-2024-04-15-04-41-52-653'
    
    run_deployment_job(training_job_name, deployment_instance_count, deployment_instance_type)