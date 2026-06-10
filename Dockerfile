FROM python:3.12-slim

WORKDIR /app

# Copy and install dependencies first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code after dependencies are installed
COPY . .

EXPOSE 8000

CMD ["bash"]
