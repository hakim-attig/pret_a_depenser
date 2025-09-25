
import unittest
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'api'))

from fastapi.testclient import TestClient
import joblib
import pandas as pd
import numpy as np
from main import app

class TestCreditScoringAPI(unittest.TestCase):
    """Tests unitaires pour l'API de scoring crédit"""
    
    def setUp(self):
        """Configuration avant chaque test"""
        self.client = TestClient(app)
        
    def test_root_endpoint(self):
        """Test de l'endpoint racine"""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("API", data)
        self.assertIn("champion", data)
        
    def test_status_endpoint(self):
        """Test du health check"""
        response = self.client.get("/status")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("api_status", data)
        self.assertEqual(data["api_status"], "online")
        
    def test_model_info_endpoint(self):
        """Test des infos modèle"""
        response = self.client.get("/model/info")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("model_type", data)
        self.assertIn("threshold", data)
        
    def test_predict_demo_valid_data(self):
        """Test prédiction avec données valides"""
        test_data = {
            "EXT_SOURCE_2": 0.6,
            "EXT_SOURCE_3": 0.7,
            "AMT_GOODS_PRICE": 300000,
            "CODE_GENDER": "M",
            "EXT_SOURCE_1": 0.5,
            "AMT_ANNUITY": 20000,
            "AMT_CREDIT": 350000,
            "DAYS_BIRTH": -12000,
            "INST_PAYMENT_PERC_mean": 0.9,
            "NAME_EDUCATION_TYPE": "Higher education"
        }
        
        response = self.client.post("/predict/demo", json=test_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("probability", data["prediction"])
        self.assertIn("decision", data["prediction"])
        
        # Vérifier que la probabilité est entre 0 et 1
        prob = data["prediction"]["probability"]
        self.assertGreaterEqual(prob, 0.0)
        self.assertLessEqual(prob, 1.0)
        
    def test_predict_demo_invalid_data(self):
        """Test avec données invalides"""
        invalid_data = {
            "AMT_CREDIT": -100000,  # Montant négatif invalide
            "AMT_GOODS_PRICE": 300000
        }
        
        response = self.client.post("/predict/demo", json=invalid_data)
        # Peut être 400 (erreur validation) ou 200 selon la logique
        self.assertIn(response.status_code, [200, 400, 422])
        
    def test_predict_demo_edge_cases(self):
        """Test cas limites"""
        edge_cases = [
            # Âge très jeune
            {
                "DAYS_BIRTH": -6570,  # 18 ans
                "AMT_CREDIT": 100000,
                "AMT_GOODS_PRICE": 120000,
                "AMT_ANNUITY": 5000,
                "CODE_GENDER": "F"
            },
            # Âge très élevé  
            {
                "DAYS_BIRTH": -29200,  # 80 ans
                "AMT_CREDIT": 50000,
                "AMT_GOODS_PRICE": 60000,
                "AMT_ANNUITY": 3000,
                "CODE_GENDER": "M"
            }
        ]
        
        for case in edge_cases:
            with self.subTest(case=case):
                response = self.client.post("/predict/demo", json=case)
                self.assertEqual(response.status_code, 200)
                data = response.json()
                self.assertIn("prediction", data)
                
    def test_business_logic_threshold(self):
        """Test que le seuil métier est appliqué"""
        # Données simulant un client très sûr
        safe_client = {
            "EXT_SOURCE_1": 0.9,
            "EXT_SOURCE_2": 0.9, 
            "EXT_SOURCE_3": 0.9,
            "AMT_GOODS_PRICE": 200000,
            "AMT_CREDIT": 150000,
            "AMT_ANNUITY": 8000,
            "DAYS_BIRTH": -11000,
            "INST_PAYMENT_PERC_mean": 1.0,
            "CODE_GENDER": "M",
            "NAME_EDUCATION_TYPE": "Higher education"
        }
        
        response = self.client.post("/predict/demo", json=safe_client)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        prob = data["prediction"]["probability"]
        decision = data["prediction"]["decision"]
        
        # Pour un client très sûr, la probabilité devrait être faible
        # mais attention au seuil de 9%
        if prob < 0.09:
            self.assertIn("FIABLE", decision)
        else:
            self.assertIn("DÉFAUT", decision)

if __name__ == "__main__":
    unittest.main()
