from prometheus_client import Counter, Histogram, Gauge

# Request metrics
REQUEST_COUNT = Counter(
    'app_request_count',
    'Application Request Count',
    ['method', 'endpoint', 'http_status']
)

REQUEST_LATENCY = Histogram(
    'app_request_latency_seconds',
    'Application Request Latency',
    ['method', 'endpoint']
)

# Prediction metrics
PREDICTION_COUNT = Counter(
    'app_prediction_count',
    'Prediction Count',
    ['prediction_class']
)

MODEL_LOAD_STATUS = Gauge(
    'app_model_load_status',
    'Model Loading Status'
)