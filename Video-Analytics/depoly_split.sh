#!/bin/bash

cd src/
docker build -t video-img .
cd ../
uid=325476609965

aws ecr delete-repository --repository-name video-img

aws ecr create-repository --repository-name video-img

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $uid.dkr.ecr.us-east-1.amazonaws.com

docker tag video-img:latest $uid.dkr.ecr.us-east-1.amazonaws.com/video-img:latest

docker push $uid.dkr.ecr.us-east-1.amazonaws.com/video-img:latest

aws lambda create-function \
--function-name Video-Analytics-stage0 \
--role arn:aws:iam::325476609965:role/lambda-url-role \
--code ImageUri=325476609965.dkr.ecr.us-east-1.amazonaws.com/video-img:latest \
--package-type Image \
--timeout 360 \
--memory-size 7168

# Update the function code
# aws lambda update-function-code \
#     --function-name Video-Analytics-stage0 \
#     --image-uri 325476609965.dkr.ecr.us-east-1.amazonaws.com/video-img:latest