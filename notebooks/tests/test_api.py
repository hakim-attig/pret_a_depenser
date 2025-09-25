import unittest
import sys
import os
from fastapi.testclient import TestClient

# Ajouter le chemin vers l'API
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

try:
    from main import app
    API_AVAILABLE = True
except ImportError as e:
    print(f"API non disponible pour tests: {e}")
    API_AVAILABLE = False

class TestCreditScoringAPI(unittest.TestCase):
    
    def setUp(self):
        if not API_AVAILABLE:
            self.skipTest("API non disponible")
        self.client = TestClient(app)
        
    def test_root_endpoint(self):
        """Test endpoint racine"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        
    def test_status_endpoint(self):
        """Test health check"""  
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        
    def test_predict_demo_basic(self):
        """Test prédiction basique"""
        test_data = {
            "AMT_GOODS_PRICE": 300000,
            "AMT_CREDIT": 250000,
            "AMT_ANNUITY": 15000,
            "DAYS_BIRTH": -12000,
            "CODE_GENDER": "M"
        }
        response = self.client.post("/predict/demo", json=test_data)
        self.assertEqual(response.status_code, 200)

if __name__ == "__main__":
    unittest.main()