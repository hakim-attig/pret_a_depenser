import unittest
import sys
import os

def run_all_tests():
    print("=== TESTS UNITAIRES PROJET SCORING CREDIT ===")
    
    # Découvrir les tests avec les vrais noms de fichiers
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Ajouter manuellement les fichiers de tests
    test_files = ['test_api', 'test_business_logic']
    
    for test_file in test_files:
        try:
            # Importer le module de test
            test_module = __import__(test_file)
            # Charger les tests du module
            module_suite = loader.loadTestsFromModule(test_module)
            suite.addTest(module_suite)
            print(f"Tests chargés depuis: {test_file}.py")
        except ImportError as e:
            print(f"Impossible de charger {test_file}.py: {e}")
    
    # Lancer les tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Résumé
    print(f"\n=== RÉSUMÉ ===")
    print(f"Tests lancés: {result.testsRun}")
    print(f"Succès: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Échecs: {len(result.failures)}")
    print(f"Erreurs: {len(result.errors)}")
    
    if result.testsRun > 0:
        success_rate = (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100
        print(f"Taux de réussite: {success_rate:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_all_tests()
    print(f"\nStatut: {'SUCCÈS' if success else 'ÉCHEC'}")