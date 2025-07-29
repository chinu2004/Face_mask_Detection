# Use official Python image
FROM python:3.10-slim

# Set working directory in container
WORKDIR /app

# Copy all local files to container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Expose port Flask will run on
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
