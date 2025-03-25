# ULog Analysis Agent API

An API for analyzing ULog files from drones/autopilots using AI-powered tools and reasoning.

## Features

- Upload and analyze ULog files (.ulg) from PX4 autopilots
- Interactive stream-based AI analysis with tool usage
- Supports multiple LLM providers (Gemini and Groq)
- Comprehensive API for building custom interfaces
- Visualizations and data processing

## Setup

### Prerequisites

- Python 3.9+
- API keys for either Google Gemini or Groq

### Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up environment variables:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Running the Server

### Development

```bash
python server.py
```

### Production

We recommend using Docker:

```bash
docker build -t ulog-analysis-agent .
docker run -p 5000:5000 --env-file .env ulog-analysis-agent
```

Or with gunicorn:

```bash
gunicorn --bind 0.0.0.0:5000 "api:app"
```

## API Documentation

Access the API documentation at: http://localhost:5000/api/docs

## Client Example

A demo client is available at http://localhost:5000/client_example.html
