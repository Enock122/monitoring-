#!/bin/bash

# ---------------------------
# 1. Create a data folder
# ---------------------------
mkdir -p /opt/render/project/src/data

# ---------------------------
# 2. Download the model from GitHub Release (if not exists)
# ---------------------------
MODEL_PATH="/opt/render/project/src/data/crop_disease_model.h5"
if [ ! -f "$MODEL_PATH" ]; then
    echo "Downloading model from GitHub Release..."
    curl -L -o "$MODEL_PATH" \
    https://github.com/Enock122/monitoring-/releases/download/v1.0/crop_disease_model.h5
fi

# ---------------------------
# 3. Download large dataset from Google Drive (if not exists)
# ---------------------------
DATASET_ZIP="/opt/render/project/src/data/plantvillage_dataset.zip"
DATASET_FOLDER="/opt/render/project/src/data/plantvillage_dataset"
FILEID="15X0JslrEzApnO1NOEbsmIUJaN77H-ALH"

if [ ! -d "$DATASET_FOLDER" ]; then
    echo "Downloading dataset from Google Drive..."
    # Handle large file confirmation
    CONFIRM=$(curl -sc /tmp/cookie "https://drive.google.com/uc?export=download&id=${FILEID}" \
      | grep -o 'confirm=[^&]*' | sed 's/confirm=//')

    curl -Lb /tmp/cookie \
      "https://drive.google.com/uc?export=download&confirm=${CONFIRM}&id=${FILEID}" \
      -o "$DATASET_ZIP"

    # Unzip the dataset
    unzip -o "$DATASET_ZIP" -d "$DATASET_FOLDER"
fi

# ---------------------------
# 4. Start Flask app
# ---------------------------
echo "Starting Flask app..."
gunicorn app:app --bind 0.0.0.0:$PORT

