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



#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":

    role, bucket, region, boto3_session, sagemaker_Sess = setup_sagemaker()
    
    model = PyTorchModel(
        entry_point="inference.py",
        role=role,
        model_data = "s3://sagemaker-ca-central-1-397567358266/models/pytorch-training-2024-04-12-20-10-24-217/output/model.tar.gz",
        framework_version = "1.13",
        py_version = "py39",
        sagemaker_session = sagemaker_Sess
    )
    
    
    predictor = model.deploy(
                            initial_instance_count = 1,
                            instance_type = "ml.t2.medium",
                            serializer = JSONSerializer(),
                            deserializer = JSONDeserializer(),
                            logs = True
                            )
    
    
    
    