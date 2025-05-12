#!/usr/bin/env bash
# Exit on error
set -e

echo "Attempting to install system dependencies for audio..."

# Try with sudo, be tolerant if sudo is not available
if command -v sudo &> /dev/null
then
    echo "Sudo found. Using sudo for apt operations."
    SUDO_CMD="sudo"
else
    echo "Sudo not found. Proceeding without sudo (might fail if root is required)."
    SUDO_CMD=""
fi

echo "Updating package lists..."
$SUDO_CMD apt-get update -y || { echo "apt-get update failed. Trying to clean and retry..."; $SUDO_CMD apt-get clean; $SUDO_CMD apt-get update -y -o Acquire::ForceIPv4=true; }

echo "Installing portaudio19-dev and libasound2-dev..."
$SUDO_CMD apt-get install -y --no-install-recommends portaudio19-dev libasound2-dev

echo "Cleaning up apt cache..."
$SUDO_CMD apt-get clean
$SUDO_CMD rm -rf /var/lib/apt/lists/*

echo "System audio dependencies installation attempt complete."