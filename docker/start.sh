#!/bin/bash

cd "$(dirname "$0")"
cd ..
workspace_dir=$PWD

if [ "$(docker ps -aq -f status=exited -f name=centerpoint)" ]; then
    docker rm ds_net;
fi

docker run -it -d --rm \
    -e "NVIDIA_DRIVER_CAPABILITIES=all" \
    --env="DISPLAY" \
    --env="QT_X11_NO_MITSHM=1" \
    --privileged \
    --name ds_net \
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
    --net "host" \
    --shm-size=512m \
    -v $workspace_dir/:/home/docker_ds_net/:rw \
    x64/ds_net:latest
