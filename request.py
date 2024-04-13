# from sagemaker.serializers import JSONSerializer
# from sagemaker.deserializers import JSONDeserializer



# # Create a JSONDeserializer
# deserializer = JSONDeserializer()

# # Example output JSON data
# output_json = {"prediction": [0.1, 0.2, 0.3, 0.4, 0.5]}

# # Deserialize the output JSON data
# serializer = JSONSerializer(content_type='application/json')
# serialized_output = serializer.serialize(data = output_json)
# # Deserialize the output JSON data
# deserializer = JSONDeserializer()
# deserialized_output = deserializer.deserialize(serialized_output, content_type = "application/json")
# print(deserialized_output)

import requests
import json
from requests_aws4auth import AWS4Auth  # Ensure you have this library installed


# Assuming endpoint_url is the URL of your deployed model endpoint
endpoint_url = "https://runtime.sagemaker.ca-central-1.amazonaws.com/endpoints/pytorch-inference-2024-04-12-23-59-03-711/invocations"

# Example input data
input_data = {"inputs": "input data"}  # Example input JSON data

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