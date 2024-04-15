from processing_job import run_processing_job, setup_sagemaker
from training_job import run_training_job
from deployment_job import run_deployment_job
from inference import make_inference

#------------------------------------------------------------------------------
# Running the script directly
if __name__ == "__main__":
    
    # Defining the reuqired parameters
    max_len = 500
    train_size = 0.8
    validation_size = 0.15
    test_size = 0.05
    embed_dim = 2
    lstm_size = 2
    bidirectional = False
    num_layers = 1
    dropout = 0
    learning_rate = 0.01
    epochs = 1
    threshold = 0.5
    batch_size = 32
    
    # The type of instance to run the job on
    processing_instacne_type = "ml.t3.large"
    # The number of instances to use for processing job
    processing_instance_count = 1
    
    train_instance_type = "ml.m5.xlarge"
    
    train_instance_count = 1
    
    deployment_instance_count = 1
    
    deployment_instance_type = "ml.t2.medium"
    
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