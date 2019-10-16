FROM nvidia/cuda:9.0-runtime
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update -qq -y\
  && apt-get install -y python3-software-properties software-properties-common libcudnn7 \
  && add-apt-repository -y ppa:deadsnakes/ppa \
  && apt-get update -qq -y \
  && apt-get install -y python3.6 python3.6-dev python3-pip python-opencv \
  && apt-get install -y git cmake build-essential ffmpeg \
  && apt-get clean \
  && rm -rf /usr/share/doc* \
  && rm -rf /var/lib/apt/lists/*

COPY requirements-cuda.txt .
#RUN conda create -y -n deepfacelab python=3.6.6 cudatoolkit=9.0 cudnn=7.3.1
#RUN conda activate deepfacelab
RUN python3.6 -m pip install -r requirements-cuda.txt
RUN python3.6 -m pip install dlib
RUN ldconfig
