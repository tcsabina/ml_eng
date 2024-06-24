import matplotlib.pyplot as plt

# Visualization libraries
import seaborn as sns

# Third-party library imports for data handling
import numpy as np

# Machine learning and model evaluation imports from scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import (
    roc_curve, auc, roc_auc_score,
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    mean_squared_error
)

class ModelTrainer:
    def __init__(
            self,
            model,
            X,
            y,
            categorical_features,
            numerical_features,
            test_size=0.3,
            random_state=42):
        """
        Initialize the ModelTrainer with a model, dataset, and preprocessing info.

        :param model: The machine learning model to be trained.
        :param X: Feature set.
        :param y: Target variable.
        :param categorical_features: List of names of the categorical features.
        :param numerical_features: List of names of the numerical features.
        :param test_size: Size of the test dataset.
        :param random_state: Seed for the random number generator.
        """
        self.model = model
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state)

        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ])

        self.pipeline = Pipeline(
            steps=[('preprocessor', self.preprocessor), ('model', model)])

    def train(self):
        """
        Train the model using the training dataset.
        """
        self.pipeline.fit(self.X_train, self.y_train)

    def evaluate(self, is_classifier=True):
        """
        Evaluate the model on the test set using specified logic and metrics, including the confusion matrix.

        :param is_classifier: Flag to indicate if the model is a classifier.
        :return: A dictionary containing evaluation metrics and the confusion matrix.
        """
        results = {}
        y_pred = self.pipeline.predict(self.X_test)

        if is_classifier:
            y_pred_proba = self.pipeline.predict_proba(
                self.X_test)[:, 1] if hasattr(self.model, "predict_proba") else None
            y_pred = [1 if x >= 0.5 else 0 for x in y_pred_proba]
            results['accuracy'] = accuracy_score(self.y_test, y_pred)
            results['precision'] = precision_score(
                self.y_test, y_pred, zero_division=0)
            results['recall'] = recall_score(
                self.y_test, y_pred, zero_division=0)
            results['f1'] = f1_score(self.y_test, y_pred, zero_division=0)

            if y_pred_proba is not None:
                results['roc_auc'] = roc_auc_score(self.y_test, y_pred_proba)

            results['confusion_matrix'] = confusion_matrix(self.y_test, y_pred)
        else:
            results['mse'] = mean_squared_error(self.y_test, y_pred)
            results['rmse'] = np.sqrt(results['mse'])

        self.results = results

        return results

    def plot_confusion_matrix(self):
        """
        Plots the confusion matrix using Seaborn's heatmap.
        """
        if 'confusion_matrix' not in self.results:
            raise ValueError(
                "No confusion matrix to plot. Please run evaluate() first.")

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            self.results['confusion_matrix'],
            annot=True,
            fmt="d",
            cmap="Blues")
        plt.title("Confusion Matrix")
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    def plot_roc_curve(self):
        """
        Plot the ROC curve for the model if it is a classifier and supports probability predictions.
        """
        if not hasattr(
                self,
                'y_test') or not hasattr(
                self,
                'model') or not hasattr(
                self.model,
                "predict_proba"):
            raise ValueError(
                "Model must be trained and be a classifier with predict_proba method before plotting ROC curve.")

        # Compute probabilities
        y_pred_proba = self.pipeline.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        # Plot
        plt.figure()
        plt.plot(
            fpr,
            tpr,
            color='darkorange',
            lw=2,
            label='ROC curve (area = %0.2f)' %
            roc_auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.show()

    def predicted_one_person(self, patient_data):
        """
        Predicts LOS for a single patient using the trained model.

         :param patient_data: A pandas DataFrame row containing features for the patient.
        """
        if not hasattr(self, 'model'):
           raise ValueError("Model must be trained before plotting.")

        # Preprocess the patient data using the same pipeline
        transformed_data = self.preprocessor.transform(patient_data)

        # Make a prediction on the transformed data
        predicted_los = self.model.predict(transformed_data)[0]

        print("Predicted Length of Stay:", predicted_los, "days")

    def plot_actual_vs_predicted(self):
        """
        Plots Actual vs. Predicted values for the regression model.
        """
        if not hasattr(self, 'y_test') or not hasattr(self, 'model'):
            raise ValueError(
                "Model must be trained and evaluated before plotting.")

        y_pred = self.pipeline.predict(self.X_test)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_test, y_pred, alpha=0.5)
        plt.plot([self.y_test.min(), self.y_test.max()], [
                 self.y_test.min(), self.y_test.max()], 'k--', lw=4)
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.title('Actual vs. Predicted')
        plt.show()

    def plot_residuals_histogram(self):
        """
        Plots a histogram of the residuals.
        """
        if not hasattr(self, 'y_test') or not hasattr(self, 'model'):
            raise ValueError(
                "Model must be trained and evaluated before plotting.")

        y_pred = self.pipeline.predict(self.X_test)
        residuals = self.y_test - y_pred

        plt.figure(figsize=(10, 6))
        sns.histplot(residuals, kde=True, edgecolor='k', color='blue')
        plt.xlabel('Residual')
        plt.ylabel('Frequency')
        plt.title('Histogram of Residuals')
        plt.show()
