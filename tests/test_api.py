from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)

def test_health_check():
    response = client.get('/health')
    assert response.status_code == 200
    assert response.json() == {'status':'active', 'model_version':"v1.0"}

def test_predictions_valid_data():

    payload = {
        "recency": 30, 
        "frequency":5,
        "monetary":100,
        "spend_velocity":1.2,
        "purchase_interval_std":5,
        "unique_products": 10,
        "unique_categories": 3,
        "weekend_ratio":0.2,
        "avg_unit_price":20,
        "whale_score":0.1,
        "return_rate":0.05,
        "avg_hour":12

    }

    response = client.post("/predict_churn", json=payload)
    assert response.status_code == 200
    assert "churn_probability" in response.json()

def test_prediction_invalid_data():
    payload = {"recency": 30}
    response = client.post("/predict_churn", json=payload)
    assert response.status_code == 422