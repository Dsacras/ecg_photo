FROM python:3.8.6-buster

COPY requirements.txt /requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
COPY ecg_model /ecg_model

CMD uvicorn ecg_model.api.api:app --host 0.0.0.0 --port $PORT
