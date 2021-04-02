#!/bin/sh

docker buildx build --push --build-arg	http_proxy=$http_proxy --build-arg https_proxy=$https_proxy --platform linux/amd64 --tag $DOCKER_REGISTRY/ml/vnet . 
