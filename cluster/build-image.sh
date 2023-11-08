#!/bin/bash
set -e
docker build . -f cluster/Dockerfile -t docker.io/damowerko/opf -t lc1.alelab:32000/owerko/opf
# push to local registry
docker push lc1.alelab:32000/owerko/opf