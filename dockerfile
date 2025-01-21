# Use the official Python image as a base
FROM python:3.9-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_PORT=8501

# Create and set the working directory in the container
WORKDIR /app

# Copy the requirements file to the container
COPY requirements.txt /app/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the Streamlit app code to the container
COPY . /app/

# Expose the default Streamlit port
EXPOSE 8501

# Set the Streamlit config to disable browser address warnings
RUN mkdir -p ~/.streamlit && \
    echo "\
[server]\n\
enableCORS=false\n\
headless=true\n\
port=$STREAMLIT_PORT\n\
" > ~/.streamlit/config.toml

# Command to run the Streamlit app
CMD ["streamlit", "run", "app.py"]
