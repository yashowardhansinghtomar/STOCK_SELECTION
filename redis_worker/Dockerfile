FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y gcc libpq-dev && rm -rf /var/lib/apt/lists/*

# 👇 Updated path — copy from root into container
COPY ../requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

# 👇 Still copy the whole app context
COPY . .

CMD ["python", "worker.py"]
