#!/bin/bash
# Create a zip of the out-spectral directory and upload it to Google Drive

# Make sure we're in the right directory
cd "$(dirname "$0")"

# Check if out-spectral directory exists
if [ ! -d "out-spectral" ]; then
  echo "Error: out-spectral directory not found."
  echo "Make sure you are running this script from the LossGeometry/models/nanoGPT directory."
  exit 1
fi

# Check if pydrive is installed
if ! pip list | grep -q pydrive; then
  echo "Installing pydrive..."
  pip install pydrive oauth2client
fi

# Check if the zip was created successfully
if [ ! -f "out-spectral.zip" ]; then
  echo "Error: Failed to create out-spectral.zip."
  exit 1
fi

# Check if the upload script exists
if [ ! -f "upload_to_drive.py" ]; then
  echo "Error: upload_to_drive.py not found."
  exit 1
fi

# Run the upload script
echo "Uploading to Google Drive..."
python upload_to_drive.py

echo "Done!" 