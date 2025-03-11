#!/bin/bash
set -e
IMAGE_NAME="opf"
DOCKER_USERNAME="damowerko"

# comma separated list of arguments, printf adds an extra comma at the end, so we remove it
printf -v args "\"%s\"," "$@"
args=${args%,}

# Get the image digest to ensure we're using the exact image we just built
IMAGE_DIGEST=$(docker inspect --format='{{index .RepoDigests 0}}' $DOCKER_USERNAME/$IMAGE_NAME:latest | cut -d'@' -f2)
echo "Using image digest: $IMAGE_DIGEST"


template=$(cat << EOF
apiVersion: batch/v1
kind: Job
metadata:
  generateName: opf-train-
  namespace: owerko
spec:
  completions: 1
  parallelism: 1
  backoffLimit: 0
  ttlSecondsAfterFinished: 3600
  template:
    spec:
      hostIPC: true
      securityContext:
        runAsUser: 1000
      restartPolicy: Never
      volumes:
      - name: opf-data
        nfs:
          server: lc1-alelab.seas.upenn.edu
          path: /nfs/general/opf_data
      containers:
      - name: opf-train
        image: docker.io/$DOCKER_USERNAME/$IMAGE_NAME@$IMAGE_DIGEST
        imagePullPolicy: Always
        command: ["python", "-u", "scripts/main.py", $args, "--simple_progress"]
        env:
        - name: WANDB_ENTITY
          value: damowerko-academic
        - name: WANDB_USERNAME
          value: damowerko
        - name: WANDB_PROJECT
          value: opf
        - name: WANDB_API_KEY
          valueFrom:
            secretKeyRef:
              name: wandb
              key: api_key
        resources:
          requests:
            cpu: 8
            memory: 32Gi
          limits:
            cpu: 64
            memory: 128Gi
            nvidia.com/gpu: 1
        volumeMounts:
        - mountPath: /home/default/opf/data
          name: opf-data
          readOnly: true
EOF
)
echo "$template" | kubectl create -f -