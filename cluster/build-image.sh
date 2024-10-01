#!/bin/bash
set -e
docker build . -t docker.io/damowerko/opf
docker push docker.io/damowerko/opf