FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgles2 libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mediapipe==1.1.1
WORKDIR /app
ADD https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task .
COPY placar.py painel.html test_placar.py ./
EXPOSE 8000
ENTRYPOINT ["python", "placar.py"]
