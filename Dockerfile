# Use the official Python image as a base
FROM python:3.12

# Set the working directory
WORKDIR /app

# Copy requirements.txt
COPY requirements.txt .

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libglib2.0-dev \
    && rm -rf /var/lib/apt/lists/*

# For fixing ImportError: libGL.so.1: cannot open shared object file: No such file or directory
RUN apt-get update
RUN apt install -y libgl1-mesa-glx

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application files
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Command to run the app
CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
