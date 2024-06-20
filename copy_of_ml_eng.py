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

# Create an instance of DataAcquisition
data_acquisition = DataAcquisition.DataAcquisition(url)

# Call download_and_extract method to download and extract the file
data_acquisition.download_and_extract()

# Define the paths to the required CSV files
file_paths = {
    'admissions': 'mimic-iii-clinical-database-demo-1.4/ADMISSIONS.csv',
    'patients': 'mimic-iii-clinical-database-demo-1.4/PATIENTS.csv',
    'callout': 'mimic-iii-clinical-database-demo-1.4/CALLOUT.csv',
    'icustays': 'mimic-iii-clinical-database-demo-1.4/ICUSTAYS.csv',
    'drgcodes': 'mimic-iii-clinical-database-demo-1.4/DRGCODES.csv',
    'services': 'mimic-iii-clinical-database-demo-1.4/SERVICES.csv'}


# Create an instance of DataFrameLoader and load the data
loader = DataFrameLoader.DataFrameLoader(file_paths)
dataframes = loader.load_dataframes()

# Instantiate DataPreprocessor()
preprocessor = DataPreprocessor.DataPreprocessor(dataframes)

# Invoke the preprocess method to perform all the preprocessing steps
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

# since 'dob' is not in datetime format, convert it first
patients_df['dobdatetime'] = pd.to_datetime(patients_df['dob'])

# Extract month from dob
patients_df['dob_month'] = patients_df['dobdatetime'].dt.month_name()

# Define the order of months
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']

# Convert month to categorical data with specified order
patients_df['dob_month'] = pd.Categorical(patients_df['dob_month'], categories=month_order, ordered=True)

# Count the number of entries for each month
entries_per_month = patients_df['dob_month'].value_counts().sort_index()

# Calculate the percentage of entries for each month
percentage_per_month = entries_per_month / len(patients_df) * 100

# Plot histogram sorted by month
plt.figure(figsize=(10, 6))
ax = entries_per_month.plot(kind='bar', color='skyblue')

# Annotate each bar with the corresponding percentage
for i, v in enumerate(percentage_per_month):
    ax.text(i, entries_per_month[i] + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)

plt.title('DOB Entries by Month')
plt.xlabel('Month')
plt.ylabel('Number of Entries')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Extract year from dobdatetime
patients_df['dob_year'] = patients_df['dobdatetime'].dt.year

# Plot histogram by year
plt.figure(figsize=(10, 6))
patients_df['dob_year'].plot(kind='hist', bins=210, color='skyblue')
plt.title('DOB Entries by Year')
plt.xlabel('Year')
plt.ylabel('Number of Entries')
plt.tight_layout()
plt.show()