# test_api.py - Tests unitaires pour l'API
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_root():
    """Test page d'accueil"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "API" in data

def test_status():
    """Test statut API"""
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["api_status"] == "online"

def test_predict_demo():
    """Test prédiction démo"""
    test_data = {
        "EXT_SOURCE_2": 0.7,
        "EXT_SOURCE_3": 0.6,
        "AMT_GOODS_PRICE": 450000,
        "CODE_GENDER": "M",
        "AMT_ANNUITY": 25000,
        "AMT_CREDIT": 500000,
        "DAYS_BIRTH": -15000
    }
    
    response = client.post("/predict/demo", json=test_data)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data

print("Tests créés avec succès")