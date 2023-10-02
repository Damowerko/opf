#!/bin/bash
args="$@"
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
          server: 158.130.113.30
          path: /nfs/general/opf_data
      containers:
      - name: opf-train
        image: docker.io/damowerko/opf
        imagePullPolicy: Always
        command: ["bash", "-c", "python -u scripts/main.py train $args && sleep 1"]
        env:
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
          limits:
            nvidia.com/gpu-12gb: 1
            cpu: 16
            memory: 12Gi
        volumeMounts:
        - mountPath: /home/default/opf/data
          name: opf-data
          readOnly: true
EOF
)
echo "$template" | kubectl create -f -