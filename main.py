from processing_job import run_processing_job, setup_sagemaker
from training_job import run_training_job
from deployment_job import run_deployment_job
from inference import make_inference

#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    

    # Maximum review text sequence length
    max_len = 500
    # Fraction of training data of all data
    train_size = 0.8
    # Fraction of validation data of all data
    validation_size = 0.15
    # Fraction of test data of all data
    test_size = 0.05
    # The type of instance to run the job on

    # Size of the embedding vector for each token
    embed_dim = 20
    # Size of the lstm output
    lstm_size = 20
    # Whether to run a bidirectional LSTM
    bidirectional = True
    # Number of LSTM layers
    num_layers = 1
    # LSTM dropout
    dropout = 0
    # Learning rate for trianing the model
    learning_rate = 0.001
    # Number of epochs to run
    epochs = 5
    # Setting the threshold for positive and negative labels
    threshold = 0.5
    # The batch size to be used during model training
    batch_size = 32
    processing_instacne_type = "ml.t3.large"
    # The number of instances to use for processing job
    processing_instance_count = 1
    # The instance type (VM type) for the training job 
    train_instance_type = "ml.m5.xlarge"
    # The number of VM nodes to use for training job
    train_instance_count = 1
    # The type of VM to use for deployment and inference    
    deployment_instance_type = "ml.t2.medium"
    # The number of VM nodes to use for deployment and inference    
    deployment_instance_count = 1
    # Whether the script is run on a local machine (True) or in a sagemaker notebook
    local = True
    
    # Setting up the sagemaker session
    role, bucket, region, boto3_session, sagemaker_Sess = setup_sagemaker(local)
    
    session_info = (role, bucket, region, boto3_session, sagemaker_Sess)
    
    print("\nData processing started\n")
    run_processing_job(session_info, processing_instacne_type, processing_instance_count,\
                           max_len, train_size, validation_size, test_size, local)
    
    print("Data processing finished, started training the model\n")
    training_job_name = run_training_job(embed_dim, lstm_size, bidirectional, num_layers, dropout, learning_rate,
                    epochs, threshold, batch_size, train_instance_type, train_instance_count, local ,session_info)
    
    print("\nModel trained and saved, deploying the model\n")
    endpoint_name = run_deployment_job(training_job_name, deployment_instance_count, deployment_instance_type, session_info)
    
    print("\nMaking some predictions with the model")
    review =  "I absoloutley love this product, it is amazing, definitely recommend you to buy."
    response = make_inference(review, endpoint_name)