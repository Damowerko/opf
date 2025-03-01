#!/bin/bash
set -e
# optional tag argument
TAG=$1
if [ -z "$TAG" ]; then
    TAG="latest"
fi

REPOSITORY="docker.io"
IMAGE_NAME="opf"
DOCKER_USERNAME="damowerko"
export DOCKER_BUILDKIT=1
docker pull $REPOSITORY/$DOCKER_USERNAME/$IMAGE_NAME:latest || true
# add tag to the image
docker tag $REPOSITORY/$DOCKER_USERNAME/$IMAGE_NAME:latest $REPOSITORY/$DOCKER_USERNAME/$IMAGE_NAME:$TAG
# build the image to ensure it is up to date
docker build . -t $REPOSITORY/$DOCKER_USERNAME/$IMAGE_NAME:$TAG --build-arg BUILDKIT_INLINE_CACHE=1  --build-arg REPOSITORY=$REPOSITORY
# push the image to the repository
docker push $REPOSITORY/$DOCKER_USERNAME/$IMAGE_NAME:$TAG