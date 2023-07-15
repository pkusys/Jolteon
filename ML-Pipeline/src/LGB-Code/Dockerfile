FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y gcc gcc-c++ make 

RUN pip3 install numpy lightgbm scikit-learn

COPY lambda_function.py   ./
COPY train.py   ./
COPY utils.py   ./
CMD ["lambda_function.handler"]      

