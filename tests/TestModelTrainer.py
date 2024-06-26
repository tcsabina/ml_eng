import unittest
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import the ModelTrainer class here
from ml_eng.ModelTrainer import ModelTrainer

class TestModelTrainer(unittest.TestCase):

    def setUp(self):
        # Set up sample data
        np.random.seed(42)
        X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
        self.X = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        self.y = y

        self.categorical_features = []  # Replace with actual categorical feature names if applicable
        self.numerical_features = list(self.X.columns)

        # Initialize ModelTrainer with a RandomForestClassifier
        self.model = RandomForestClassifier(random_state=42)
        self.trainer = ModelTrainer(
            model=self.model,
            X=self.X,
            y=self.y,
            categorical_features=self.categorical_features,
            numerical_features=self.numerical_features
        )

    def test_model_training(self):
        # Train the model
        self.trainer.train()

        # Assert that the model attribute is set
        self.assertTrue(hasattr(self.trainer, 'model'))

    def test_model_evaluation(self):
        # Train the model (if not already trained)
        self.trainer.train()

        # Evaluate the model
        evaluation_results = self.trainer.evaluate()

        # Check if evaluation results contain expected metrics
        self.assertIn('accuracy', evaluation_results)
        self.assertIn('precision', evaluation_results)
        self.assertIn('recall', evaluation_results)
        self.assertIn('f1', evaluation_results)

        # Assert that accuracy is within a reasonable range
        self.assertGreaterEqual(evaluation_results['accuracy'], 0.0)
        self.assertLessEqual(evaluation_results['accuracy'], 1.0)

    def test_predict_one_person(self):
        # Train the model (if not already trained)
        self.trainer.train()

        # Example of patient data (should match the format of your training data)
        patient_data = pd.DataFrame([[0.5, -1.0, 1.2, 0.8, -0.3]], columns=self.X.columns)

        # Make prediction for the patient
        try:
            self.trainer.predicted_one_person(patient_data)
        except ValueError as e:
            self.fail(f"predicted_one_person raised ValueError unexpectedly: {str(e)}")

if __name__ == '__main__':
    unittest.main()
