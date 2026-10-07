# CareOps-AI Architecture

```text
Synthetic data generation
        ↓
PostgreSQL / SQLAlchemy persistence
        ↓
FastAPI API layer
        ↓
Forecasting + optimization services
        ↓
Operational recommendations + alerts
        ↓
What-if simulation
        ↓
AI explanation layer
        ↓
React + TypeScript dashboard
        ↓
Audit and run history
```

## Components

- Synthetic data generator: produces reproducible hospital operational datasets.
- PostgreSQL layer: stores source, forecast, optimization, and audit outputs.
- FastAPI backend: exposes health, patient, forecast, optimization, recommendation, simulation, and audit endpoints.
- Model services: reuse the existing ML pipeline and optimizer without duplicating its logic.
- Frontend: renders a portfolio-ready dashboard based on real API responses.
- AI assistant: provides explanation using structured model output, not direct forecasting.

## Limitations

This project remains synthetic and is not a live clinical decision system.
