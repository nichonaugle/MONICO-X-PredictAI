# Use the official Python image as the base image
FROM python:3.11-slim

# Set the working directory in the container
WORKDIR /src

# Copy the requirements file and install dependencies
COPY requirements.txt /src/

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Python project into the container
COPY . /src/

# Expose the port that FastAPI will use (e.g., port 8000)
EXPOSE 80

# Command to run FastAPI with Uvicorn (and mDNS service registration if possible)
CMD uvicorn app:app --host=0.0.0.0 --port=80