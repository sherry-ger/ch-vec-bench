FROM python:3.13-slim

WORKDIR /app

# Install dependencies first (layer caching)
COPY pyproject.toml .
COPY src/ src/
RUN pip install -e . -q

# /data is the mount point for user-supplied vector files
VOLUME ["/data"]

ENTRYPOINT ["python", "-m", "src.main"]
CMD ["--help"]
