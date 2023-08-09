FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y gcc gcc-c++ make wget

RUN yum install -y libEGL libGL libX11 libXext libXfixes libdrm

RUN yum install -y tkinter && yum install -y python3-tkinter

RUN rm /var/lang/bin/python && ln -s /usr/bin/python3.7 /var/lang/bin/python

RUN pip3.7 install urllib3==1.26.7

RUN pip3.7 install torch torchvision tensorflow-cpu imageai opencv-python

RUN pip3.7 install awslambdaric

RUN rm /var/lang/bin/python3.10 && ln -s /usr/bin/python3.7 /var/lang/bin/python3.10

COPY lambda_function.py   ./
COPY utils.py   ./
COPY split.py ./
COPY utils.py ./
COPY extract.py ./
COPY preprocess.py ./
COPY classify.py ./
COPY tiny-yolov3.pt ./
CMD ["lambda_function.handler"]      