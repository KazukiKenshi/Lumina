FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY best_emotion_model.pth .
COPY api_server.py .
COPY model.py . 

EXPOSE 8001

CMD ["python", "api_server.py"]