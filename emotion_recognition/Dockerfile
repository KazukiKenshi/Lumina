FROM python:3.10-slim

WORKDIR /app

# Install dependencies needed for PyTorch CPU wheels
RUN apt-get update && apt-get install -y \
    gcc \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install CPU-only PyTorch (small)
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision torchaudio

RUN pip install --no-cache-dir -r requirements.txt

COPY best_emotion_model.pth .
COPY api_server.py .
COPY model.py .

EXPOSE 8001

CMD ["python", "api_server.py"]


