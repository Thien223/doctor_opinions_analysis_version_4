#/bin/bash
sudo docker rm docker_train || true && sudo docker load -i docker_train_img.taz && sudo docker build --rm -t itazthien/doctor_opinions_train:lastest . && sudo docker volume create train_volume