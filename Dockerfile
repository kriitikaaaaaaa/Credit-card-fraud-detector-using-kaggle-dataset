# Base image (replace with a suitable Python image)
FROM python:3.10

# Set working directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install -r requirements.txt  

# Expose port (if your API uses a specific port)
EXPOSE 5000  

# Command to run the application
CMD ["python", "app.py"] 