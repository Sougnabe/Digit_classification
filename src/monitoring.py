from prometheus_client import Counter, Histogram

REQUEST_COUNT = Counter("api_requests_total", "Total prediction requests")
REQUEST_LATENCY = Histogram("api_request_latency_seconds", "Prediction request latency")
RETRAIN_COUNT = Counter("model_retrains_total", "Total retraining runs")
