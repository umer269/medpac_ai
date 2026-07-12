# ── Build stage ───────────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /app

# Install system dependencies for scipy / nibabel
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Muhammad Umer Raja <umer.raja@medpacs.dev>"
LABEL description="MedPACS-AI: Clinical-grade DICOM ETL pipeline with MRIQC QC and research-backed normalization"
LABEL version="1.0.0"

# Runtime system libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY etl/        ./etl/
COPY dashboard/  ./dashboard/
COPY api/        ./api/
COPY config.yaml .
COPY main.py     .

# Create directories for runtime data
RUN mkdir -p /data/dicom_inbox /output/nifti /output/previews

# Non-root user for security
RUN useradd -m -u 1000 medpacs && chown -R medpacs:medpacs /app /data /output
USER medpacs

# Environment
ENV PYTHONUNBUFFERED=1
ENV MEDPACS_DATA_DIR=/data
ENV MEDPACS_OUTPUT_DIR=/output

EXPOSE 5000 8000

# Default: run the ETL pipeline
CMD ["python", "main.py", "--input-dir", "/data/dicom_inbox"]
