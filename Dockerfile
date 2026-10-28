FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend ./backend
COPY scripts ./scripts
COPY data ./data

# Build the index once at image build time for the sample/demo site.
# In production, mount a real data/index volume and run the nightly
# sync as a separate cron job / scheduled task instead (see README).
RUN python -m scripts.build_index || true

EXPOSE 8000
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
