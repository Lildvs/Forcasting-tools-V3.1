# Forecasting Tools Metrics API

This document provides documentation for the Metrics API, which allows you to access and analyze forecasting metrics data programmatically.

## Overview

The Metrics API exposes a REST interface for calculating, storing, and retrieving forecasting metrics such as Brier score, calibration, coverage, and peer score. The API provides endpoints for:

- Calculating metrics on batches of prediction data
- Storing prediction datasets for future analysis
- Retrieving stored datasets and results
- Analyzing metrics across different models and questions

## Getting Started

### Running the API Server

1. Install the required dependencies:
   ```bash
   pip install -r api_requirements.txt
   ```

2. Start the API server:
   ```bash
   python api_server.py
   ```

3. Run both the API server and Streamlit app together:
   ```bash
   ./run_with_api.sh
   ```

4. The API will be available at `http://localhost:8000`, and the Swagger UI documentation at `http://localhost:8000/docs`.

### Converting Existing Data

To convert existing backtest data to the API format:

```bash
python scripts/convert_metrics_data.py --group-by model_name
```

## API Endpoints

### Health Check

```
GET /api/health
```

Returns the health status of the API.

### Metrics Calculation

```
POST /api/metrics/calculate
```

Calculate metrics for a batch of predictions and outcomes.

**Request Body:**
```json
{
  "predictions": [0.7, 0.3, 0.5, 0.9, 0.1],
  "outcomes": [1, 0, 1, 1, 0],
  "metric_type": "brier_score",
  "save_result": true,
  "dataset_name": "My Predictions",
  "metadata": {}
}
```

**Response:**
```json
{
  "metric_name": "brier_score",
  "value": 0.24,
  "timestamp": "2023-09-23T15:30:45.123456",
  "metadata": {
    "data_points": 5,
    "dataset_name": "My Predictions",
    "dataset_id": "dataset_20230923153045_5"
  }
}
```

### Dataset Management

```
GET /api/metrics/datasets
```

List all stored datasets.

```
GET /api/metrics/datasets/{dataset_id}
```

Get a specific dataset by ID.

```
POST /api/metrics/datasets
```

Create a new dataset.

```
DELETE /api/metrics/datasets/{dataset_id}
```

Delete a dataset.

### Calculating from Dataset

```
POST /api/metrics/calculate_from_dataset/{dataset_id}?metric_type=brier_score&save_result=true
```

Calculate metrics using data from a stored dataset.

## Data Models

### MetricsResult

```json
{
  "metric_name": "string",
  "value": 0.0,
  "timestamp": "2023-09-23T15:30:45.123456",
  "metadata": {}
}
```

### MetricsDataset

```json
{
  "dataset_id": "string",
  "name": "string",
  "predictions": [0.7, 0.3, 0.5, 0.9, 0.1],
  "outcomes": [1, 0, 1, 1, 0],
  "confidence_intervals": [[0.6, 0.8], [0.2, 0.4], [0.3, 0.7], [0.8, 1.0], [0.0, 0.2]],
  "model_names": ["model_A", "model_A", "model_A", "model_A", "model_A"],
  "question_ids": ["q1", "q2", "q3", "q4", "q5"],
  "created_at": "2023-09-23T15:30:45.123456",
  "last_updated": "2023-09-23T15:30:45.123456",
  "metadata": {}
}
```

## Examples

### Calculate Brier Score

```python
import requests
import json

# Define the data
data = {
    "predictions": [0.7, 0.3, 0.5, 0.9, 0.1],
    "outcomes": [1, 0, 1, 1, 0],
    "metric_type": "brier_score",
    "save_result": True,
    "dataset_name": "Example Dataset"
}

# Send the request
response = requests.post(
    "http://localhost:8000/api/metrics/calculate",
    json=data
)

# Print the result
result = response.json()
print(f"Brier Score: {result['value']}")
```

### Calculate Peer Score

```python
import requests
import json

# Define the data
data = {
    "predictions": [0.8, 0.2, 0.9, 0.7, 0.1, 0.6, 0.4, 0.7, 0.6, 0.3, 0.4, 0.6, 0.5, 0.4, 0.5],
    "outcomes": [1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0],
    "metric_type": "peer_score",
    "save_result": True,
    "dataset_name": "Multiple Models Comparison",
    "metadata": {
        "model_names": ["model_A", "model_A", "model_A", "model_A", "model_A", 
                       "model_B", "model_B", "model_B", "model_B", "model_B",
                       "model_C", "model_C", "model_C", "model_C", "model_C"]
    }
}

# Send the request
response = requests.post(
    "http://localhost:8000/api/metrics/calculate",
    json=data
)

# Print the result
result = response.json()
print(f"Peer Scores: {result['metadata']['detailed_result']}")
```

## Error Handling

The API uses standard HTTP status codes:

- `200 OK`: The request succeeded
- `400 Bad Request`: The request was invalid (e.g., missing required parameters)
- `404 Not Found`: The requested resource was not found
- `500 Internal Server Error`: An error occurred on the server

Error responses include a JSON body with a `detail` field describing the error.

## Security Considerations

This API is designed for local or internal use and does not include authentication. If deploying to a public environment, add appropriate authentication and authorization mechanisms, and restrict CORS to trusted domains. 