#!/usr/bin/env bash
# Exit on error
set -e
echo "Installing system dependencies for audio..."
apt-get update -y
apt-get install -y portaudio19-dev libasound2-dev
echo "System audio dependencies installation complete."
