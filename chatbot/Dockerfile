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

COPY src /app/src
COPY api_server.py .
COPY data /app/data


EXPOSE 5001

CMD ["python", "api_server.py"]
