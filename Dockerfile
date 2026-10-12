FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg

WORKDIR /app

COPY requirements-lock.txt /app/
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements-lock.txt

COPY . /app

# Run as an unprivileged user; it owns /app so the pipeline can rewrite outputs.
RUN useradd --create-home --uid 10001 careops && chown -R careops:careops /app
USER careops

EXPOSE 8000

HEALTHCHECK --interval=10s --timeout=5s --start-period=20s --retries=5 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]
