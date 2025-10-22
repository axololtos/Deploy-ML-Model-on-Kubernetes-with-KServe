FROM python:3.9-slim
WORKDIR /app

# Install required packages
RUN pip install flask scikit-learn joblib numpy

# Copy model files
COPY model/ ./model/

# Copy the Flask app
COPY app.py .

EXPOSE 8080
CMD ["python", "app.py"]