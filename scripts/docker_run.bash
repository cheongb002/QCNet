#!/bin/bash

DRIVE=$1

VOLUMES="--volume=${PWD}/assets:/home/$(whoami)/assets
        --volume=${PWD}/src:/home/$(whoami)/src
        --volume=${PWD}/results:/home/$(whoami)/results
        --volume=${DRIVE}:/home/$(whoami)/data"

GPU='"device=0"'

docker run \
-it \
-p 6006:6006 \
--privileged \
-e DISPLAY=unix$DISPLAY \
-e NVIDIA_DRIVER_CAPABILITIES=all \
-v /tmp/.X11-unix/:/tmp/.X11-unix/ \
--gpus $GPU \
--shm-size 32G \
$VOLUMES \
--name=qcnet \
qcnet
