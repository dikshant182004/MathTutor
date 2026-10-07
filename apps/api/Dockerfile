FROM python:3.13-slim
WORKDIR /app
COPY requirements.txt /app/requirements.txt
COPY requirements-test.txt /app/requirements-test.txt
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY . /app
ENV PYTHONPATH=/app/src
EXPOSE 8000
CMD ["uvicorn","apps.api.main:app","--host","0.0.0.0","--port","8000"]
