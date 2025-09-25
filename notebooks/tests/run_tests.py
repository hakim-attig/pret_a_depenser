import unittest
import sys
import os

def run_all_tests():
    print("=== TESTS UNITAIRES PROJET SCORING CREDIT ===")
    
    # Découvrir tous les tests
    loader = unittest.TestLoader()
    suite = loader.discover('.', pattern='test_*.py')
    
    # Lancer les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print(f"\n=== RÉSUMÉ ===")
    print(f"Tests: {result.testsRun}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / max(result.testsRun, 1) * 100
    print(f"Taux de réussite: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    print(f"\nStatut: {'SUCCÈS' if success else 'ÉCHEC'}")