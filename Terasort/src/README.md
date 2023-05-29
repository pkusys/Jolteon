# Create Lambda funtion with C++
## Install dependency
```
sudo apt-get update
sudo apt-get install -y libcurl4-openssl-dev libssl-dev uuid-dev zlib1g-dev libpulse-dev
sudo apt install -y cmake build-essential libssl-dev pkg-config
sudo apt-get install -y zlib1g-dev
sudo apt install -y awscli
# assume that cmake has been installed
```
## Install AWS lambda environment
```
cd ~
git clone https://github.com/awslabs/aws-lambda-cpp.git
cd aws-lambda-cpp
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
   -DCMAKE_INSTALL_PREFIX=~/lambda
make && make install
cd -
```
## Install AWS SDK for C++
```
cd ~
git clone --recurse-submodules https://github.com/aws/aws-sdk-cpp
cd aws-sdk-cpp
mkdir build
cd build
cmake .. -DBUILD_ONLY=s3 \
 -DBUILD_SHARED_LIBS=OFF \
 -DENABLE_UNITY_BUILD=ON \
 -DCMAKE_BUILD_TYPE=Release \
 -DCMAKE_INSTALL_PREFIX=~/lambda

make && make install
cd -
```
## Build function zip
```
mkdir build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=~/lambda
make
make aws-lambda-package-main
```

## Create the function
```
aws lambda create-function \
--function-name terasort-1 \
--role arn:aws:iam::325476609965:role/lambda-url-role \
--runtime provided \
--timeout 15 \
--memory-size 128 \
--handler encoder \
--zip-file fileb://main.zip
```

## Update the function
```
aws lambda update-function-code \
    --function-name terasort-1 \
    --zip-file fileb://main.zip
```
