# Use the official Python image as the base image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY /src .
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000
EXPOSE 5353
# Command to run the application
CMD ["python", "src/main.py"]