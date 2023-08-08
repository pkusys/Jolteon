FROM public.ecr.aws/lambda/python:3.10

RUN yum install -y gcc gcc-c++ make wget

RUN yum install -y libEGL libGL libX11 libXext libXfixes libdrm

# install epel-release
RUN rpm -ivh https://dl.fedoraproject.org/pub/epel/epel-release-latest-7.noarch.rpm
RUN rpm --import http://li.nux.ro/download/nux/RPM-GPG-KEY-nux.ro
RUN rpm -Uvh http://li.nux.ro/download/nux/dextop/el7/x86_64/nux-dextop-release-0-1.el7.nux.noarch.rpm

RUN yum install -y libwayland-client

RUN wget http://mirror.centos.org/centos/7/os/x86_64/Packages/libva-1.8.3-1.el7.x86_64.rpm && rpm -i libva-1.8.3-1.el7.x86_64.rpm

RUN yum install -y ffmpeg

RUN pip3 install numpy moviepy

COPY lambda_function.py   ./
COPY utils.py   ./
COPY split.py ./
RUN hahah
CMD ["lambda_function.handler"]      