FROM ubuntu:14.04
ARG TENSORFLOW_VERSION=r1.4

ENV CI_BUILD_PYTHON python
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH
ENV TF_CUDA_VERSION 8.1
ENV TF_CUDNN_VERSION 7
ENV TF_NEED_CUDA 1
ENV TF_CUDA_COMPUTE_CAPABILITIES=3.7,7.0

RUN apt-get update && apt-get install -y \
    git \
    wget \
    nano \
    python-pip python-dev \
 

# pip
RUN pip install --upgrade pip

   
RUN \
    cd deep_patient_mimiciii_multitask/ && \
    pip install -r requirements.txt && \
