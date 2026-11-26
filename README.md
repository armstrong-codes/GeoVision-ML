# GeoVision ML
### Satellite Image Classification Using Deep Learning

## 1. About the Project

GeoVision ML is a machine learning project that classifies satellite images into different land-use categories. It uses the EuroSAT RGB dataset, which contains 27,000 satellite images across 10 classes.

The goal of this project is to build a complete application where users can upload satellite images, get predictions, and view model performance through a simple dashboard.

## 2. Technologies Used

- Python – Main programming language
- TensorFlow/Keras – Building and training models
- Pandas and NumPy – Data processing
- Scikit-learn – Model evaluation
- FastAPI – Backend API
- Streamlit – User dashboard
- Docker – Running the application in containers
- Locust – Testing API performance
- Microsoft Azure – Cloud deployment

## 3. Machine Learning Models

Two models were developed and compared:

- **Custom CNN:** A convolutional neural network built for classifying satellite images.
- **MobileNetV2:** A pre-trained model adapted for satellite image classification.

Both models are evaluated using accuracy, precision, recall, F1-score, and a confusion matrix.

## 4. Dataset Classes

The EuroSAT dataset contains 10 categories:

1. AnnualCrop
2. Forest
3. HerbaceousVegetation
4. Highway
5. Industrial
6. Pasture
7. PermanentCrop
8. Residential
9. River
10. SeaLake

## 5. Project Structure

```text
GeoVision-ML/
├── api/
├── dashboard/
├── data/
├── models/
├── notebook/
├── src/
├── monitoring/
├── locust/
├── results/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## 6. Running the Project Locally

First, clone the repository:

```bash
git clone https://github.com/armstrong-codes/GeoVision-ML.git
cd GeoVision-ML
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Start the API:

```bash
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Start the dashboard in another terminal:

```bash
python -m streamlit run dashboard/app.py --server.port 8501
```

Open the application:

- Dashboard: `http://localhost:8501`
- API documentation: `http://localhost:8000/docs`

## 7. Application Features

The dashboard provides:

- An overview of the application.
- Satellite image upload and prediction.
- Uploading additional training data.
- Model retraining.
- Model performance information.

The FastAPI backend handles predictions and other application requests.

## 8. Load Testing

Locust was used to test how the API performs with different numbers of containers.

The tests used 10 users, a spawn rate of 2 users per second, and a duration of 2 minutes.

| API containers | Requests | Failures | Average response time | P95 |
|---:|---:|---:|---:|---:|
| 1 | 572 | 0 | 59 ms | 170 ms |
| 2 | 576 | 0 | 41 ms | 110 ms |
| 3 | 574 | 0 | 84 ms | 230 ms |

The two-container setup performed best in these tests, with the lowest average and P95 response times. Increasing the number of containers did not significantly improve throughput.

Load-testing results and comparison graphs are stored in `results/load_testing/`.

## 9. Cloud Deployment

The project is being deployed using Microsoft Azure Container Apps, with Docker images stored in GitHub Container Registry.

- **API:** https://geovision-api.politesky-a35ce14a.southafricanorth.azurecontainerapps.io
- **Dashboard:** https://geovision-dashboard.politesky-a35ce14a.southafricanorth.azurecontainerapps.io
- **API documentation:** https://geovision-api.politesky-a35ce14a.southafricanorth.azurecontainerapps.io/docs?utm_source=chatgpt.com#/default/get_retraining_status_retrain_status_get
**Demo video:** ADD VIDEO LINK

## 10. Conclusion

GeoVision ML brings together machine learning, backend development, and cloud deployment to solve a real-world image classification problem. Through this project, we developed and evaluated deep learning models and built an application that allows users to interact with them.