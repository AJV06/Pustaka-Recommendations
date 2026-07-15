# Use official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Prevent Python from writing pyc files to disc
ENV PYTHONDONTWRITEBYTECODE 1
# Prevent Python from buffering stdout and stderr
ENV PYTHONUNBUFFERED 1

# Install dependencies
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the backend code
COPY backend/app /app/app
COPY backend/.env /app/.env

# The user explicitly asked to be able to deploy without code changes. 
# We need the data and models directories included in the Docker image 
# for a fully self-contained deployment to Render/Railway.
COPY data /app/data
COPY models /app/models

# Expose port
EXPOSE 8000

# Set environment variables for the container
ENV PORT=8000
ENV MODEL_PATH=/app/models
ENV DATA_PATH=/app/data

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
