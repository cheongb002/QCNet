#!/bin/bash

sudo bash -c "echo '' > $(docker inspect --format="{{.LogPath}}" qcnet)"
docker attach --detach-keys="ctrl-a" qcnet