FROM ubuntu:20.04

RUN apt-get clean -y && apt-get update -y && apt-get -y install locales

RUN sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && \
    locale-gen
ENV LANG en_US.UTF-8  
ENV LANGUAGE en_US:en  
ENV LC_ALL en_US.UTF-8

ENV DEBIAN_FRONTEND noninteractive

RUN apt-get install -y --no-install-recommends build-essential g++ git make python3 python3-dev python3-minimal python3-numpy python3-pip python3-setuptools python3-venv software-properties-common sudo unzip vim wget curl

RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#nvidia tensorrt-pytorch requirements
RUN pip3 install notebook absl-py>=0.7.0 scipy sphinx_glpi_theme prettytable pyyaml tqdm ninja cython ipywidgets

COPY pytorch-quantization /etc/pytorch-quantization

RUN python3 /etc/pytorch-quantization/setup.py install

COPY bashrc /etc/bashrc

RUN touch /root/.bashrc \
 && cat /etc/bashrc >> /root/.bashrc

ADD banner.sh /etc/banner.sh

