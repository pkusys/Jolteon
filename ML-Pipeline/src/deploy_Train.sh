#!/bin/bash

cd LGB-Code/
docker build -t lgb-img .
cd ../
uid=325476609965

aws ecr delete-repository --repository-name lgb-img

aws ecr create-repository --repository-name lgb-img

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $uid.dkr.ecr.us-east-1.amazonaws.com

docker tag lgb-img:latest $uid.dkr.ecr.us-east-1.amazonaws.com/lgb-img:latest

docker push $uid.dkr.ecr.us-east-1.amazonaws.com/lgb-img:latest

aws lambda create-function \
--function-name ML-Pipeline-stage1 \
--role arn:aws:iam::325476609965:role/lambda-url-role \
--code ImageUri=325476609965.dkr.ecr.us-east-1.amazonaws.com/lgb-img:latest \
--package-type Image \
--timeout 360 \
--memory-size 2048

# Update the function code
# aws lambda update-function-code \
#     --function-name ML-Pipeline-stage1 \
#     --image-uri 325476609965.dkr.ecr.us-east-1.amazonaws.com/lgb-img:latest