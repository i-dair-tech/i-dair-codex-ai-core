FROM python:3.9.15
RUN apt update -y && \
    apt install -y curl nano && \
    mkdir -p /usr/src/app/log /usr/src/app/mlruns /usr/src/dataset && \
    chmod -R 777 /usr/src/app /usr/src/dataset
WORKDIR /usr/src/app
COPY . ./
RUN pip install -r requirements.txt && \
    groupadd -r node && useradd -r -g node node
USER node:node