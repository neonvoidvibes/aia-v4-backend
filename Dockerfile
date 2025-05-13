# Use an official Python runtime
FROM python:3.11-slim

# bust cache so static ffmpeg always re-downloads
ARG CACHEBUST=1

WORKDIR /app

RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
      portaudio19-dev \
      libasound2-dev \
      build-essential \
      git \
      curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    \
    # install ffmpeg 6.x static build
    && curl -fsSL https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz \
       | tar -xJ --wildcards --strip-components=1 -C /usr/local/bin \
         '*/ffmpeg' '*/ffprobe'

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

# print ffmpeg version to logs, then launch WITH 1 WORKER using gthread
CMD ["sh", "-c", "ffmpeg -version && ffprobe -version && exec gunicorn --worker-class gthread --workers 1 --threads 4 --bind 0.0.0.0:${PORT} api_server:app --log-file - --error-logfile - --access-logfile - --log-level info --timeout 120"]