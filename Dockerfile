# Use Python 3.9 slim base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements files for both training and prediction
COPY training/requirements.txt training_requirements.txt
COPY prediction/requirements.txt prediction_requirements.txt

# Install dependencies
RUN pip install --no-cache-dir -r training_requirements.txt -r prediction_requirements.txt

# Copy training and prediction scripts
COPY training/train.py /app/train.py
COPY prediction/predict.py /app/predict.py

# Set the entrypoint
ENTRYPOINT ["python"]

