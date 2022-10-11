#1.  First Function named: serializeImageData

#Test

{
  "image_data": "",
  "s3_bucket": "sagemaker-us-east-1-298735464366", 
  "s3_key": "test/bicycle_s_000513.png"
}


import json
import boto3
import base64

s3 = boto3.client('s3')

def lambda_handler(event, context):
    """A function to serialize target data from S3"""
    
    # Get the s3 address from the Step Function event input
    key = event["s3_key"]## TODO: fill in
    bucket = event["s3_bucket"]## TODO: fill in
    
    # Download the data from s3 to /tmp/image.png
    ## TODO: fill in 
    s3.download_file(bucket,key,'/tmp/image.png')
    # We read the data from a file
    with open("/tmp/image.png", "rb") as f:
        image_data = base64.b64encode(f.read())

    # Pass the data back to the Step Function
    print("Event:", event.keys())
    return {
        'statusCode': 200,
        'body': {
            "image_data": image_data,
            "s3_bucket": bucket,
            "s3_key": key,
            "inferences": []
        }
        
    }

#Answer

{
  "statusCode": 200,
  "body": {
    "image_data": "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",
    "s3_bucket": "sagemaker-us-east-1-298735464366",
    "s3_key": "test/bicycle_s_000513.png",
    "inferences": []
  }
}

#---------------------------------------------------------

#2.1  Second Function named: classifyImageData

#Test
{
  "image_data": "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",
  #"s3_bucket": "sagemaker-us-east-1-298735464366",
  #"s3_key": "test/bicycle_s_000513.png",
  #"inferences": []
}

#Handler  this example helped me https://youtu.be/-iU36P8hizs
import json
#import sagemaker
import base64
#from sagemaker.serializers import IdentitySerializer
import boto3

# Fill this in with the name of your deployed model
ENDPOINT ='image-classification-2022-10-10-19-34-48-961' ## TODO: fill in
runtime=boto3.Session().client('sagemaker-runtime')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    #predictor = Predictor(ENDPOINT)## TODO: fill in

    # For this model the IdentitySerializer needs to be "image/png"
    #predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    #inferences = predictor.predict(payload)## TODO: fill in
    
    response=runtime.invoke_endpoint(EndpointName=ENDPOINT,ContentType='image/png',Body=image)
    
    result=json.loads(response['Body'].read().decode())
    # We return the data back to the Step Function    
    #event["inferences"] = inferences.decode('utf-8')
    return {
        'statusCode': 200,
        'body': json.dumps(result)
    }


#Answer

Response
{
  "statusCode": 200,
  "body": "[1.0, 0.0]"
}


#2.2 

import json
#import sagemaker
import base64
#from sagemaker.serializers import IdentitySerializer
import boto3

# Fill this in with the name of your deployed model
ENDPOINT ='image-classification-2022-10-10-19-34-48-961' ## TODO: fill in
runtime=boto3.Session().client('sagemaker-runtime')

def lambda_handler(event, context):

    # Decode the image data
    image = base64.b64decode(event['image_data'])

    # Instantiate a Predictor
    #predictor = Predictor(ENDPOINT)## TODO: fill in

    # For this model the IdentitySerializer needs to be "image/png"
    #predictor.serializer = IdentitySerializer("image/png")
    
    # Make a prediction:
    #inferences = predictor.predict(payload)## TODO: fill in
    
    response=runtime.invoke_endpoint(EndpointName=ENDPOINT,ContentType='image/png',Body=image)
    
    # We return the data back to the Step Function    
    #event["inferences"] = inferences.decode('utf-8')
    event["inferences"]=json.loads(response['Body'].read().decode('utf-8'))
    return {
        'statusCode': 200,
        'body': json.dumps(event["inferences"])
    }

# Answer

Response
{
  "statusCode": 200,
  "body": "[1.0, 0.0]"
}



#-----------------------------------------------------------------------

#3. Third Lambda Function

{
  "body": "[1.0, 0.0]"
}

import json


THRESHOLD = .93


def lambda_handler(event, context):
    
    # Grab the inferences from the event
    inferences = event["body"]
    
    # Check if any values in our inferences are above THRESHOLD
    inferences=inferences.strip("")
    inferences=inferences[1:-1].split(',')

    for item in inferences:
        if float(item)>0.93:
            meets_threshold = float(item) ## TODO: fill in
        

    
    # If our threshold is met, pass our data back out of the
    # Step Function, else, end the Step Function with an error
    if meets_threshold:
        pass
    else:
        raise("THRESHOLD_CONFIDENCE_NOT_MET")

    return {
        'statusCode': 200,
        'body': json.dumps(event)
    }

#Answer

Response
{
  "statusCode": 200,
  "body": "{\"body\": \"[1.0, 0.0]\"}"
}