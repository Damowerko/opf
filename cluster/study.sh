#!/bin/bash
set -e
IMAGE_NAME="opf"
DOCKER_USERNAME="damowerko"

# comma separated list of arguments, printf adds an extra comma at the end, so we remove it
printf -v args "\"%s\"," "$@"
args=${args%,}

# build first
$(dirname "$0")/build.sh

template=$(cat << EOF
apiVersion: batch/v1
kind: Job
metadata:
  name: opf-study
  namespace: owerko
spec:
  completions: 400
  parallelism: 8
  ttlSecondsAfterFinished: 28800  
  template:
    spec:
      restartPolicy: Never
      volumes:
      - name: opf-data
        nfs:
          server: lc1-alelab.seas.upenn.edu
          path: /nfs/general/opf_data
      containers:
      - name: opf-study
        image: docker.io/$DOCKER_USERNAME/$IMAGE_NAME
        imagePullPolicy: Always
        command: ["bash", "-c", "python -u scripts/main.py study $args --simple_progress && sleep 1"]
        env:
        - name: OPTUNA_STORAGE
          value: postgresql://optuna:optuna@optuna-db.owerko.svc.cluster.local:5432/optuna
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
            memory: 16Gi
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
