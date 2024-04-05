#!/bin/bash

DIR=$(dirname "$0")
DIR=${DIR%/}

docker build --no-cache \
    --build-arg USERNAME=$(whoami) \
    -t qcnet \
    -f $DIR/../docker/Dockerfile \
    $DIR/../docker