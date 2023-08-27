FROM python:3.10

COPY ./requirements.txt /app/

WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3-distutils

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD [ "python3", "./app.py" ]