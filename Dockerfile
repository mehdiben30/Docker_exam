# Use a Python base image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Copy requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project files into the container
COPY . .

# Expose any ports if needed (e.g., for APIs, not necessary here)
# EXPOSE 8080

# Define the default command to run the train-classifier script
CMD ["python", "train-classifier.py"]
