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


# Initialize the DataVisualizer with your processed dataframe
visualizer = DataVisualizer.DataVisualizer(patients_df)

# To plot patient outcomes by admission type
visualizer.plot_patient_outcomes_by_admission_type()

# To plot average length of stay by admission type and insurance
visualizer.plot_average_los_by_admission_and_insurance()

# Function to calculate age
def calculate_age(row):
    dob = pd.to_datetime(row["dob"]).to_pydatetime()
    admittime = pd.to_datetime(row["admittime"]).to_pydatetime()
    age = (admittime - dob).days // 365
    return age


patients_df["age"] = patients_df.apply(calculate_age, axis=1)
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
X = patients_df[features]
y = patients_df["hospital_expire_flag"]

# Pipeline for transformations
numerical_features = ["age", "los"]

categorical_features = [
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "ethnicity",
    "last_careunit",
]

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
categorical_features = [
    "gender",
    "admission_type",
    "admission_location",
    "insurance",
    "ethnicity",
    "last_careunit",
]
# In the following we want to pre-process data to build a RandomForest regressor
# to estimate the Length of Stay (los) of patients in the hospital

# Splitting the dataset ensuring 'los' is used as target
# features should be defined as before, excluding 'los'
X = patients_df[features]
y = patients_df["los"].values  # Directly accessing 'los' values


model_trainer = ModelTrainer.ModelTrainer(RandomForestRegressor(),
                             X, y,
                             categorical_features,
                             numerical_features)
model_trainer.train()
# Assuming evaluate method adapts based on is_classifier flag
model_trainer.evaluate(is_classifier=False)
model_trainer.plot_actual_vs_predicted()
model_trainer.plot_residuals_histogram()


print("Number of entries in the dataframe:", patients_df.shape[0])

# Extract month from admittime
print(type(patients_df['admittime']))
patients_df['admittimedate'] = pd.to_datetime(patients_df['admittime'])

# Extract the month and create 'admittime_month' column
patients_df['admittime_month'] = patients_df['admittimedate'].dt.month_name() # Create the column with month names

# Define the order of months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Convert month to categorical data with specified order
patients_df['admittime_month'] = pd.Categorical(patients_df['admittime_month'], categories=month_order, ordered=True)

# Plot histogram sorted by month
plt.figure(figsize=(10, 6))
patients_df['admittime_month'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Admissions by Month')
plt.xlabel('Month')
plt.ylabel('Number of Admissions')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
