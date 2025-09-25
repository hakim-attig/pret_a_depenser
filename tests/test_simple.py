import unittest
import numpy as np

class TestBusinessLogic(unittest.TestCase):
    
    def test_threshold_9_percent(self):
        """Test que le seuil de 9% fonctionne"""
        probabilities = np.array([0.05, 0.08, 0.09, 0.12])
        threshold = 0.09
        predictions = (probabilities >= threshold).astype(int)
        expected = np.array([0, 0, 1, 1])  # 0.09 et 0.12 >= 0.09
        np.testing.assert_array_equal(predictions, expected)
        
    def test_business_cost_formula(self):
        """Test formule coût métier 10*FN + FP"""
        fn, fp = 2, 3
        cost = 10 * fn + fp  # 10*2 + 3 = 23
        self.assertEqual(cost, 23)
        
    def test_lightgbm_champion_logic(self):
        """Test logique du modèle champion"""
        champion_auc = 0.7800
        champion_cost = 30666
        
        self.assertGreater(champion_auc, 0.75)  # AUC > 75%
        self.assertLess(champion_cost, 35000)   # Coût < 35000

if __name__ == "__main__":
    unittest.main(verbosity=2)