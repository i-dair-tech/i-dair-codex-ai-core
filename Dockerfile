FROM python:3.9.15
RUN apt update -y
RUN apt install curl -y
RUN apt install nano -y
RUN mkdir /usr/src/app
RUN mkdir /usr/src/dataset
RUN chmod 777 -R /usr/src/dataset
RUN chmod 777 -R /usr/src/app
WORKDIR /usr/src/app
COPY . ./
COPY requirements.txt ./
RUN pip install  -r requirements.txt
RUN groupadd -r node && useradd -r -g node node
USER node:node



        
      

  
