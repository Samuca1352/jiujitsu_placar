FROM python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgles2 libegl1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir mediapipe==1.1.1
WORKDIR /app
# versões fixas (com checksum): os limites dos gestos foram ajustados nestes modelos
ADD --checksum=sha256:fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1 \
    https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task .
ADD --checksum=sha256:59929e1d1ee95287735ddd833b19cf4ac46d29bc7afddbbf6753c459690d574a \
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task .
COPY placar.py painel.html test_placar.py ./
EXPOSE 8000
ENTRYPOINT ["python", "placar.py"]
