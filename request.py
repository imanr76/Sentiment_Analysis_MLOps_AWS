import requests
import json
from requests_aws4auth import AWS4Auth  # Ensure you have this library installed
from dotenv import load_dotenv
import os

load_dotenv()
aws_access_key = os.getenv("AWS_PUBLIC_KEY")
aws_secret_key = os.getenv("AWS_PRIVATE_KEY")
aws_region = os.getenv("AWS_REGION")


endpoint_name = "pytorch-inference-2024-04-13-18-40-11-251"

endpoint_url="https://runtime.sagemaker.ca-central-1.amazonaws.com/endpoints/" + endpoint_name + "/invocations"

# Example input data
input_data = {"input_text": "I love it"}  # Example input JSON data

# Convert input data to JSON string
input_json = json.dumps(input_data)

# Create AWS authentication object
auth = AWS4Auth(aws_access_key, aws_secret_key, aws_region, "sagemaker")

# Send a POST request to the endpoint URL with authentication headers
response = requests.post(endpoint_url, data=input_json, auth=auth)

# Check if the request was successful
if response.status_code == 200:
    # Get the prediction result
    result = response.json()
    print("Prediction result:", result)
else:
    print("Failed to get prediction. Status code:", response.status_code)