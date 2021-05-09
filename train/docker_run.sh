#/bin/bash
sudo docker rm docker_train || true && sudo docker build --rm -t train:lastest . && echo "Done"
