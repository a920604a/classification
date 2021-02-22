FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel

LABEL maintainer="a920604a@gmail.com"

ENV DEBIAN_FRONTEND noninteractive
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
RUN set -x \
	&& apt-get update \
	&& apt-get install --no-install-recommends --no-install-suggests -y libsm6 libxrender1 libxext6 libgl1-mesa-glx libglib2.0-dev
RUN pip install numpy sklearn opencv-python 

RUN set -x 
RUN apt-get update

RUN apt-get install --no-install-recommends --no-install-suggests -y libmysqlclient-dev

RUN pip install SQLAlchemy pymysql

CMD ["/workspace/yuan-algorithm/image-classification/run.sh"]

