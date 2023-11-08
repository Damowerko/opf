#!/bin/bash
set -e
docker build . -f cluster/Dockerfile -t docker.io/damowerko/opf
docker push docker.io/damowerko/opf