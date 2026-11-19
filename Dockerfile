# syntax=docker/dockerfile:1
#
# IronLedger Analytics — production image for Azure Container Apps (Consumption plan).
#
# Build (Container Apps runs linux/amd64; always pass --platform, especially on Apple Silicon):
#   docker build --platform linux/amd64 -t ironledger-analytics:local .
# Run:
#   docker run --rm -p 8000:8000 ironledger-analytics:local
#
# The image is self-contained: code, Streamlit theme and the committed Parquet datasets are
# baked in. Nothing is written to disk at runtime, so the container also runs with a
# read-only root filesystem (see README, "Container image").

# Debian bookworm (glibc 2.36): every pinned dependency installs from a manylinux_2_28
# wheel, so no compiler toolchain is needed. Pin to a digest for fully reproducible builds.
FROM python:3.11-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    IRONLEDGER_DATA_SOURCE=local \
    IRONLEDGER_DATA_DIR=/app/data \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_FILE_WATCHER_TYPE=none \
    STREAMLIT_SERVER_RUN_ON_SAVE=false \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

# Dependencies first so this layer stays cached when only code or data change.
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

# Unprivileged runtime user. Everything under /app stays owned by root and is not
# writable by this user, so code and data are read-only at runtime.
RUN groupadd --system --gid 10001 ironledger \
 && useradd --system --uid 10001 --gid ironledger --no-create-home --home-dir /nonexistent \
        --shell /usr/sbin/nologin ironledger

# Explicit allow-list (no `COPY . .`): only what the running app needs.
COPY .streamlit/config.toml .streamlit/config.toml
COPY assets/ assets/
COPY ironledger/ ironledger/
COPY views/ views/
COPY app.py .
# Listed one by one so the build FAILS if any dataset is missing from the build context.
COPY data/dealerships.parquet \
     data/salespeople.parquet \
     data/customers.parquet \
     data/inventory.parquet \
     data/sales.parquet \
     data/operating_expenses.parquet \
     data/metadata.json \
     data/

# Build-time smoke test: load every table exactly as the app does and fail the build if
# any file is missing/corrupt or row counts disagree with metadata.json. Then make /app
# read-only for everyone (root-owned, no write bits).
RUN python -c "from ironledger.data_loader import load_dataset; \
ds = load_dataset(); \
counts = {n: len(df) for n, df in ds.tables().items()}; \
assert counts == ds.metadata['row_counts'], counts; \
print('dataset OK:', counts)" \
 && chmod -R a-w /app

USER ironledger:ironledger

EXPOSE 8000

# Local/Docker health signal. Azure Container Apps ignores HEALTHCHECK; configure its
# probes against the same path (GET /_stcore/health on port 8000).
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/_stcore/health', timeout=4).status == 200 else 1)"]

# Exec form: Streamlit is PID 1 and receives SIGTERM directly when a replica stops.
CMD ["python", "-m", "streamlit", "run", "app.py", \
     "--server.address=0.0.0.0", "--server.port=8000", "--server.headless=true"]
