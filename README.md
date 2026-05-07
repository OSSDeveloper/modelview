# Model View Project

A web application for visualizing data models with FastAPI backend and Next.js + D3.js frontend.

## Architecture

- **Backend**: FastAPI (Python) - REST API
- **Frontend**: Next.js + D3.js (TypeScript) - Interactive visualizations
- **Deployment**: Docker Compose on prd_vps

## Quick Start

```bash
# Development
docker-compose up --build

# Access
# Frontend: http://localhost:3000
# Backend: http://localhost:8000
# API Docs: http://localhost:8000/docs
```

## Project Structure

```
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   └── models/
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   ├── app/
│   ├── components/
│   ├── tests/
│   └── package.json
├── docker-compose.yml
└── deploy/
```