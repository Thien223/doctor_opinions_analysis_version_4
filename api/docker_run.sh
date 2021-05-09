#/bin/bash
sudo docker rm docker_api || true && sudo docker build --rm -t api:lastest . && echo "Done"
