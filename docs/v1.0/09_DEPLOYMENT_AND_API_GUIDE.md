# EquiMind v1.0: Deployment, API & CLI Guide

EquiMind v1.0 provides multiple entrypoints: FastAPI REST server, CLI runner, and production Docker containerization.

---

## ⚡ FastAPI REST Endpoints (`equimind.api.server`)

Launch FastAPI server:
```bash
python3 -m uvicorn equimind.api.server:app --host 0.0.0.0 --port 8000
```

### Endpoints
- `GET /api/v1/health`: System health & provider status check.
- `POST /api/v1/research`: Run equity research on a ticker.
  - **Request Body**:
    ```json
    {
      "ticker": "NVDA",
      "query": "Should I invest in NVIDIA today for AI growth?",
      "provider": "openai",
      "as_of_date": "2024-01-01"
    }
    ```
- `GET /api/v1/memory/{ticker}`: Fetch persistent entity knowledge record.

---

## 💻 CLI Runner (`equimind.cli`)

Run terminal research queries:
```bash
# General query with OpenAI
python3 -m equimind.cli --ticker NVDA --query "Analyze NVDA" --provider openai

# Backtest query with cutoff date
python3 -m equimind.cli --ticker TSLA --query "Analyze Tesla" --provider mock --as-of-date 2024-01-01
```

---

## 🐳 Production Containerization (`Dockerfile`, `docker-compose.yml`)

Build and launch with Docker Compose:
```bash
docker-compose up --build -d
```

Check container logs:
```bash
docker logs -f equimind_server
```
