FROM ubuntu:20.04

MAINTAINER itaz "thien@itaz.kr, smno@itaz.kr, yjgwak@itaz.kr"

RUN apt-get update -y && apt-get install -y python3-pip python3.8-dev

RUN update-alternatives --install /usr/local/bin/python python /usr/bin/python3.8 13

RUN DEBIAN_FRONTEND=noninteractive apt-get -yq install openjdk-14-jdk 

RUN pip3 install --upgrade pip

RUN mkdir /app

COPY . /app

WORKDIR /app

RUN pip3 install -r requirements.txt -f https://download.pytorch.org/whl/torch_stable.html

ENTRYPOINT ["python", "-u", "source_odin/train.py"]
