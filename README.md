# DigitSight: End-to-End Digit Classification Pipeline

## 1. Project Description
This repository demonstrates a full machine learning lifecycle for non-tabular data (images):
- Data acquisition and preprocessing
- Offline model training and testing
- Model serialization and API deployment
- Single-image prediction
- Bulk upload + retraining trigger
- Monitoring and dashboard UI
- Flood-request simulation with Locust
- Dockerized deployment and horizontal API scaling support

The implementation uses image data and satisfies the requirement to avoid tabular-only datasets.

Dataset provenance: the images used in this project are digit-class images (classes 0 to 9), stored locally in the repository under `data/train` and `data/test`.
The source reference used for this local dataset is the scikit-learn Digits dataset: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html
How this source was identified: class folders are `0` to `9` and image files are named like `digit_0_0000.png`, which matches a digits-export workflow from scikit-learn Digits into PNG files.

Train/test split used for all reported results: the project uses the current split already present in `data/train` (training set) and `data/test` (test set).

## 2. Repository Structure
```text
Project_name/
├── README.md
├── notebook/
│   └── project_name.ipynb
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── prediction.py
│   ├── retrain.py
│   ├── api.py
│   └── monitoring.py
├── data/
│   ├── train/
│   ├── test/
│   └── uploads/
├── models/
│   ├── image_classifier.pkl
│   └── labels.txt
├── ui/
│   └── app.py
├── locust/
│   └── locustfile.py
├── nginx/
│   └── nginx.conf
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

## 3. Setup and Run
### 3.1 Install dependencies
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 3.2 Train and evaluate offline
```bash
python -m src.model
```
This writes:
- `models/image_classifier.pkl`
- `models/labels.txt`

### 3.3 Run API
```bash
uvicorn src.api:app --reload --host 0.0.0.0 --port 8000
```

### 3.4 Run UI
```bash
streamlit run ui/app.py
```

## 4. API Endpoints
- `GET /health`: model uptime/availability health check
- `GET /metrics`: Prometheus metrics
- `POST /predict`: predict one image file
- `POST /upload-bulk?class_name=<name>&auto_retrain=<true|false>`: upload multiple images for retraining
- `POST /retrain?epochs=<int>`: manual retraining trigger
- `GET /evaluate`: production evaluation using current trained model on test set

## 5. UI Features (Rubric Coverage)
The Streamlit dashboard provides:
- Model uptime and service health
- Monitoring metrics view
- Visualizations for at least 3 dataset features:
  - Class distribution
  - Width/height distribution
  - Brightness distribution
- Single image prediction
- Bulk image upload for retraining
- Manual retraining trigger button

## 6. Notebook Coverage
Notebook path: `notebook/project_name.ipynb`

Notebook includes:
- Detailed preprocessing workflow
- Model training
- Evaluation with 5 metrics (accuracy, precision, recall, F1, log-loss)
- Classification report and confusion matrix
- Prediction helper function
- Interpretation story for at least 3 image features

## 7. Flood Request Simulation (Locust)
### 7.1 Local run (already executed)
Command:
```bash
python -m locust -f locust/locustfile.py --host http://localhost:8000 --headless -u 20 -r 5 -t 20s --only-summary
```

Observed output summary:
- Requests: 273
- Failures: 0
- Avg latency: 547.27 ms
- P95 latency: 5100 ms
- Throughput: 16.96 req/s

Save the Locust console output screenshot in your report as evidence.

### 7.2 Verifiable flood-test evidence (1,2,3 runs)
Executed command:
```bash
python src/load_test.py --skip-docker --users 10 --spawn-rate 5 --run-time 10s --replicas "1" "2" "3"
```

Final results table (from generated artifacts):
| Run Label | Users | Spawn Rate | Duration | Requests | Failures | Avg Latency (ms) | P95 (ms) | Throughput (req/s) | Log Evidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 10 | 5 | 10s | 11 | 0 | 2355 | 5000 | 1.82 | [locust_replicas_1.log](evidence/flood-test/locust_replicas_1.log) |
| 2 | 10 | 5 | 10s | 9 | 0 | 1629 | 3600 | 2.39 | [locust_replicas_2.log](evidence/flood-test/locust_replicas_2.log) |
| 3 | 10 | 5 | 10s | 18 | 0 | 1831 | 5000 | 2.99 | [locust_replicas_3.log](evidence/flood-test/locust_replicas_3.log) |

Machine-readable summary: [evidence/flood-test/summary.csv](evidence/flood-test/summary.csv)

Note: these runs were executed in local mode (`--skip-docker`) because Docker scaling was not available at execution time. The proof remains explicit and verifiable through the linked raw Locust logs and CSV summary.

## 8. Deployment

### 8.1 Render deployment
This repository includes a Render blueprint at [render.yaml](render.yaml) that defines:
- `ml-summative-api`
- `ml-summative-ui`

Deployment steps:
1. Push the repository to GitHub.
2. Connect the repository to Render.
3. Deploy from [render.yaml](render.yaml).
4. Set the UI service `API_URL` to the public URL of the API service.

Recommended public endpoints after deployment:
- API: `/health`, `/predict`, `/evaluate`, `/metrics`
- UI: Streamlit dashboard page
- Live UI URL: https://ml-summative-ui.onrender.com

### 8.2 Docker deployment and scaling
If you want to run locally or on your own infrastructure:

#### Build and run
```bash
docker compose up --build
```

#### Scale API replicas
```bash
docker compose up -d --build --scale api=3
```

Architecture:
- `api` service runs FastAPI model server (scalable)
- `gateway` (nginx) exposes host port 8000 and load-balances to API containers
- `ui` service provides the web dashboard on port 8501

### 8.3 Public deployment note
Render is the preferred public deployment path for this submission because it is simple to reproduce from GitHub. If the free tier sleeps after inactivity, the first request may be slow.

## 9. Model Evaluation Snapshot
Latest offline training run:
- Accuracy: 0.9528
- Precision: 0.9525
- Recall: 0.9528
- F1: 0.9524
- Loss: 0.1609

