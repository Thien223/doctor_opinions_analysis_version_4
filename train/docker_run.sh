#/bin/bash
sudo docker rm docker_train || true && sudo docker build --rm -t train:lastest . && sudo docker volume create train_volume