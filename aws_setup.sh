# install aws cli v2
curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
unzip awscliv2.zip
sudo ./aws/install

# configure aws cli with your own key and secret
aws configure
# check the configuration
aws configure ls

# install aws boto3 for python
pip3 install boto3

