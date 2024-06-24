# Third-party library imports for data handling
import pandas as pd

# Visualization libraries
import matplotlib.pyplot as plt

# Machine learning and model evaluation imports from scikit-learn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor

# Own classes
import DataAcquisition
import DataFrameLoader
import DataPreprocessor
import DataVisualizer
import ModelTrainer


# URL of the file to be downloaded
url = "https://physionet.org/static/published-projects/mimiciii-demo/mimic-iii-clinical-database-demo-1.4.zip"

# Create an instance of DataAcquisition and download and extract the demo data
data_acquisition = DataAcquisition.DataAcquisition(url)
data_acquisition.download_and_extract()


# Create an instance of DataFrameLoader and load the data
loader = DataFrameLoader.DataFrameLoader()
dataframes = loader.load_dataframes()


# Create an instance of DataPreprocessor and invoke the preprocess method to perform all the preprocessing steps
preprocessor = DataPreprocessor.DataPreprocessor(dataframes)
patients_df = preprocessor.preprocess()


# Create an instance of DataVisualizer and show plots
visualizer = DataVisualizer.DataVisualizer(patients_df)
#visualizer.plot_patient_outcomes_by_admission_type() # for Requirement 3
#visualizer.plot_average_los_by_admission_and_insurance() # for Requirement 1


# Prepare for model training
features = [
    "age",
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "ethnicity",
    "last_careunit",
    "los",
]
categorical_features = [
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "ethnicity",
    "last_careunit",
]
numerical_features = ["age", "los"]
X = patients_df[features]

y = patients_df["hospital_expire_flag"]
# Train the model on the basis of X and target y prepared above
model_trainer = ModelTrainer.ModelTrainer(LogisticRegression(),
                             X, y,
                             categorical_features,
                             numerical_features)
model_trainer.train()
metrics = model_trainer.evaluate(is_classifier=True)
print(metrics)
model_trainer.plot_confusion_matrix()
model_trainer.plot_roc_curve()


# Make sure 'los' is not listed in numeric_features
numerical_features = ["age"]

# In the following we want to pre-process data to build a RandomForest regressor
# to estimate the Length of Stay (los) of patients in the hospital

# Splitting the dataset ensuring 'los' is used as target
# features should be defined as before, excluding 'los'
y = patients_df["los"].values  # Directly accessing 'los' values

model_trainer = ModelTrainer.ModelTrainer(RandomForestRegressor(),
                             X, y,
                             categorical_features,
                             numerical_features)
model_trainer.train()
# Assuming evaluate method adapts based on is_classifier flag
metrics = model_trainer.evaluate(is_classifier=False)
#print(metrics)
model_trainer.plot_actual_vs_predicted()
model_trainer.plot_residuals_histogram()

patient_data_dict = {
    "age": 45,
    "gender": "Female",
    "admission_type": "emergency",
    "admission_location": "transfer from hosp/extram",
    "insurance": "medicare",
    "ethnicity": "black/african american",
    "last_careunit": "micu",
    "los": 0,
}

# Convert the dictionary to a single-row DataFrame
patient_data = pd.DataFrame(patient_data_dict, index=[0])

# Call the function to predict LOS for this patient
model_trainer.predicted_one_person(patient_data)
