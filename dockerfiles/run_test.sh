#!/bin/bash
IMAGE_NAME=drltetris:latest

docker run -it --rm --runtime=nvidia --entrypoint /bin/bash $IMAGE_NAME
