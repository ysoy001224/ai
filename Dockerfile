FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-nanum \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=10000
EXPOSE 10000

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-10000} --workers 1 --timeout 180 app:app"]
