FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1

# system deps
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# install python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# COPY EVERYTHING from root (this includes server.py + static images)
COPY . .

# ensure writable filesystem area (same folder approach you use)
RUN chmod -R 755 /app

EXPOSE 5000

# your app uses root folder for uploads + static files
ENV UPLOAD_FOLDER=/app
ENV PUBLIC_URL_BASE=http://python-service:5000
ENV CORS_ORIGINS=*

CMD ["python", "server.py"]