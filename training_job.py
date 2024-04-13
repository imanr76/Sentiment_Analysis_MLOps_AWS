from dotenv import load_dotenvimport boto3import sagemakerfrom sagemaker.pytorch.estimator import PyTorch as PytorchEstimatorimport osdef setup_sagemaker(local = True):    """    Sets up the sagemaker and boto3 sessions required for running the processing job.    Parameters    ----------    local : boolean        Whether the script is running locally or inside sagemaker notebooks.    Returns    -------    role : str, obj        ARN role for the sagemaker session.    bucket : str        The default bucket name for the sagemaker session.    region : str        Teh region of the sgaemaker and boto3 session.    boto3_session : obj    sagemaker_Sess : obj    """    # IF running the script locally    if local:        load_dotenv()        role = os.getenv("ROLE")        profile_name  = os.getenv("AWS_PROFILE")        boto3_session = boto3.session.Session(profile_name = profile_name)        sagemaker_Sess = sagemaker.Session(boto_session = boto3_session)    # If running the code from a sagemaker notebook    else:        boto3_session = boto3.session.Session()        sagemaker_Sess = sagemaker.Session()        role = sagemaker_Sess.get_execution_role()        region = sagemaker_Sess.boto_region_name    bucket = sagemaker_Sess.default_bucket()        return role, bucket, region, boto3_session, sagemaker_Sessdef run_training_job(embed_dim = 20, lstm_size = 20, bidirectional = True,                num_layers = 1, dropout = 0.0, learning_rate = 0.001,                epochs = 100, threshold = 0.5, train_instance_type = "ml.m5.xlarge",                train_instance_count = 1, local = True):        role, bucket, region, boto3_session, sagemaker_Sess = setup_sagemaker(local)        hyperparameters = {                        "embed_dim" : embed_dim,                        "lstm_size" : lstm_size,                        "bidirectional" : bidirectional,                        "num_layers" : num_layers,                        "dropout" : dropout,                        "learning_rate" : learning_rate,                        "epochs" : epochs,                        "threshold" : threshold                        }        data_channels = {"train" : "s3://" + bucket + "/data/training/",                     "validation" : "s3://" + bucket + "/data/validation/",                     "test" : "s3://" + bucket + "/data/test/",                     "vocabulary" : "s3://" + bucket + "/models/vocabulary/"}            estimator = PytorchEstimator(                                entry_point = "training.py",                                framework_version = "1.13",                                py_version = "py39",                                role = role,                                instance_count = train_instance_count,                                instance_type = train_instance_type,                                hyperparameters = hyperparameters,                                input_mode = 'File',                                output_path = "s3://" + bucket + "/models/",                                sagemaker_session = sagemaker_Sess                                )            estimator.fit(inputs = data_channels, logs = True)    #------------------------------------------------------------------------------# Running the script directlyif __name__ == "__main__":        # Size of the embedding vector for each token    embed_dim = 20    # Size of the lstm output    lstm_size = 20    # Whether to run a bidirectional LSTM    bidirectional = True    # Number of LSTM layers    num_layers = 1    # LSTM dropout    dropout = 0    # Learning rate for trianing the model    learning_rate = 0.001    # Number of epochs to run    epochs = 5    # Setting the threshold for positive and negative labels    threshold = 0.5        train_instance_type = "ml.m5.xlarge"        train_instance_count = 1        local = True        run_training_job(embed_dim, lstm_size, bidirectional, num_layers, dropout, learning_rate,                    epochs, threshold, train_instance_type, train_instance_count, local)