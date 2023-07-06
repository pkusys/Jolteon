# Create Lambda funtion with C++
## Install dependency
```
mkdir package
pip install --target ./package boto3
cd package
zip -r ../my_deployment_package.zip .
cd ..
zip my_deployment_package.zip *.py
```

## Create the function
```
aws lambda create-function \
--function-name tpcds-dsq95-stage0 \
--role arn:aws:iam::325476609965:role/lambda-url-role \
--runtime python3.10 \
--timeout 360 \
--memory-size 2048 \
--handler lambda_function.lambda_handler \
--zip-file fileb://my_deployment_package.zip
```

## Update the function code
```
aws lambda update-function-code \
    --function-name tpcds-dsq95-stage0 \
    --zip-file fileb://my_deployment_package.zip
```
