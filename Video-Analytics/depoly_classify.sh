#!/bin/bash

cd src/
docker build -t video-img-classify -f ./classify.Dockerfile .
cd ../
uid=325476609965

aws ecr delete-repository --repository-name video-img-classify

aws ecr create-repository --repository-name video-img-classify

aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin $uid.dkr.ecr.us-east-1.amazonaws.com

docker tag video-img-classify:latest $uid.dkr.ecr.us-east-1.amazonaws.com/video-img-classify:latest

docker push $uid.dkr.ecr.us-east-1.amazonaws.com/video-img-classify:latest

aws lambda create-function \
--function-name Video-Analytics-stage3 \
--role arn:aws:iam::325476609965:role/lambda-url-role \
--code ImageUri=325476609965.dkr.ecr.us-east-1.amazonaws.com/video-img-classify:latest \
--package-type Image \
--timeout 360 \
--memory-size 1792

# Update the function code
# aws lambda update-function-code \
#     --function-name Video-Analytics-stage3 \
#     --image-uri 325476609965.dkr.ecr.us-east-1.amazonaws.com/video-img-classify:latest