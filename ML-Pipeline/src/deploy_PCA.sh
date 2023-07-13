#!/bin/bash

# First, create a bucket on S3 to use as the host for the deployment packages
# This script takes that bucket name as the first input
# Second, create an execution role in AWS console. The execution role needs Full Lambda Access permissions
# Make sure upload Digits_Train.txt to Your_Bucket_Name/ML_Pipeline/Digits_Train.txt
# and upload Digits_Test.txt to Your_Bucket_Name/ML_Pipeline/Digits_Test.txt
# This script takes that execution role ARN as the second input

cd PCA/; mkdir package; pip install --target ./package numpy; 
cd package; zip -r ../../PCA_Package.zip .; cd ../
zip ../PCA_Package.zip ./*; cd ../

aws lambda create-function \
--function-name ML-Pipeline-stage0 \
--role arn:aws:iam::325476609965:role/lambda-url-role \
--runtime python3.10 \
--timeout 360 \
--memory-size 2048 \
--handler lambda_function.lambda_handler \
--zip-file fileb://PCA_Package.zip

# Update the function code
# aws lambda update-function-code \
#     --function-name ML-Pipeline-stage0 \
#     --zip-file fileb://PCA_Package.zip

echo "PCA function deployed correctly with ARN:"
pca_arn=$(aws lambda get-function --function-name ML-Pipeline-stage0 | grep Arn | grep PCA | awk -F'Arn\": "' '{print $2}' | awk -F'"' '{print $1}')
echo $pca_arn