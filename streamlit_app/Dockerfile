FROM python:3.10

COPY ./requirements.txt /app/

WORKDIR /app

RUN apt-get update
RUN apt-get install -y python3-distutils

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT ["streamlit", "run", "Homepage.py", "--server.port=8501", "--server.address=0.0.0.0"]