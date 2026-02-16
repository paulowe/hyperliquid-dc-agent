# images/dc-vae-base/Dockerfile
FROM python:3.11-slim

RUN pip install --upgrade pip setuptools wheel \
 && pip install \
     Cython \
     google-cloud-aiplatform==1.24.1 \
     kfp==2.0.0

WORKDIR /app
ENTRYPOINT ["python"]
