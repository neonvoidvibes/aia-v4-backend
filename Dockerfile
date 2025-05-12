# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PortAudio, build tools, and other potential needs
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    portaudio19-dev \
    libasound2-dev \
    build-essential \
    ffmpeg \
    git \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
# Use --no-cache-dir to reduce image size
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container
COPY . .

# Make port available to the world outside this container
# Render will set the PORT env var, Gunicorn will bind to it.
# This EXPOSE is more for documentation / local Docker runs.
EXPOSE 10000

# Define environment variable for the Gunicorn port (Render provides $PORT)
# Gunicorn will use $PORT if set, otherwise default to 8000.
# No need to explicitly set ENV PORT here as Render injects it.

# Run gunicorn when the container launches
# The $PORT variable will be injected by Render.
# Using sh -c to allow environment variable expansion in the CMD.
CMD ["sh", "-c", "gunicorn --workers ${GUNICORN_WORKERS:-4} --bind 0.0.0.0:${PORT} api_server:app --log-file - --error-logfile - --access-logfile - --log-level info"]