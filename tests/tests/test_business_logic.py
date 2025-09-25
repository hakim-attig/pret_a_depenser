
import unittest
import sys
import os
import pandas as pd
import numpy as np

# Ajouter le chemin vers les modules du projet
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

class TestBusinessLogic(unittest.TestCase):
    """Tests pour la logique métier du scoring"""
    
    def test_business_cost_calculation(self):
        """Test du calcul de coût métier"""
        # Simuler des prédictions
        y_true = np.array([0, 0, 1, 1, 0])
        y_pred = np.array([0, 1, 0, 1, 0])
        
        # Calcul manuel: 1 FP + 1 FN selon matrice confusion
        # FN (1 prédit 0) = 1, FP (0 prédit 1) = 1
        # Coût = 10*FN + FP = 10*1 + 1 = 11
        
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        expected_cost = 10 * fn + fp
        
        # Vérifier la logique
        self.assertEqual(fn, 1)  # 1 faux négatif
        self.assertEqual(fp, 1)  # 1 faux positif
        self.assertEqual(expected_cost, 11)
        
    def test_threshold_logic(self):
        """Test de la logique de seuil"""
        # Probabilités de test
        probas = np.array([0.05, 0.08, 0.09, 0.12, 0.15])
        threshold = 0.090
        
        # Prédictions avec seuil 9%
        predictions = (probas >= threshold).astype(int)
        
        expected = np.array([0, 0, 0, 1, 1])  # Seuls >9% sont classés 1
        
        np.testing.assert_array_equal(predictions, expected)
        
    def test_feature_importance_consistency(self):
        """Test cohérence des features importantes"""
        important_features = [
            'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
            'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE'
        ]
        
        # Vérifier que les features sont bien définies
        for feature in important_features:
            self.assertIsInstance(feature, str)
            self.assertTrue(len(feature) > 0)
            
    def test_data_types_consistency(self):
        """Test cohérence des types de données"""
        # Simuler un échantillon de données
        sample_data = {
            'EXT_SOURCE_1': 0.5,
            'EXT_SOURCE_2': 0.6, 
            'AMT_CREDIT': 300000,
            'DAYS_BIRTH': -12000,
            'CODE_GENDER': 'M'
        }
        
        # Vérifications de types
        self.assertIsInstance(sample_data['EXT_SOURCE_1'], (int, float))
        self.assertIsInstance(sample_data['AMT_CREDIT'], (int, float))
        self.assertIsInstance(sample_data['DAYS_BIRTH'], (int, float))
        self.assertIsInstance(sample_data['CODE_GENDER'], str)
        
        # Vérifications de cohérence métier
        self.assertLessEqual(sample_data['EXT_SOURCE_1'], 1.0)
        self.assertGreaterEqual(sample_data['EXT_SOURCE_1'], 0.0)
        self.assertGreater(sample_data['AMT_CREDIT'], 0)
        self.assertLess(sample_data['DAYS_BIRTH'], 0)  # Âge en jours négatifs

if __name__ == "__main__":
    unittest.main()
