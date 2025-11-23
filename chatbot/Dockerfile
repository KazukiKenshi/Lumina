FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src /app/src
COPY api_server.py .
COPY data /app/data


EXPOSE 5001

CMD ["python", "api_server.py"]