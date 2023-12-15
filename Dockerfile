FROM pytorch/pytorch:latest
  
ADD . /

WORKDIR /
cp sources.list /etc/apt/sources.list
 
WORKDIR /YOLOX
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN apt-get update
RUN apt install build-essential libgl1-mesa-dev libglib2.0-dev -y
RUN pip3 install -U pip && pip3 install -r requirements.txt
RUN pip3 install -v -e .
 
WORKDIR /
