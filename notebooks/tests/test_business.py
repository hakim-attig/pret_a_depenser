import unittest
import numpy as np

class TestBusinessLogic(unittest.TestCase):
    
    def test_threshold_logic(self):
        """Test logique de seuil"""
        probabilities = np.array([0.05, 0.08, 0.09, 0.12])
        threshold = 0.09
        predictions = (probabilities >= threshold).astype(int)
        expected = np.array([0, 0, 0, 1])
        np.testing.assert_array_equal(predictions, expected)
        
    def test_cost_calculation(self):
        """Test calcul de coût métier"""
        # Simuler FN=2, FP=3
        fn, fp = 2, 3
        cost = 10 * fn + fp  # Formule 10*FN + FP
        self.assertEqual(cost, 23)
        
    def test_data_validation(self):
        """Test validation des données"""
        valid_gender = "M"
        valid_credit = 300000
        valid_age_days = -12000
        
        self.assertIn(valid_gender, ["M", "F", "XNA"])
        self.assertGreater(valid_credit, 0)
        self.assertLess(valid_age_days, 0)

if __name__ == "__main__":
    unittest.main()