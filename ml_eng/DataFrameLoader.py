import pandas as pd

# Define the paths to the required CSV files
file_paths = {
    'admissions': 'data\mimic-iii-clinical-database-demo-1.4/ADMISSIONS.csv',
    'callout': 'data\mimic-iii-clinical-database-demo-1.4/CALLOUT.csv',
    'drgcodes': 'data\mimic-iii-clinical-database-demo-1.4/DRGCODES.csv',
    'icustays': 'data\mimic-iii-clinical-database-demo-1.4/ICUSTAYS.csv',
    'patients': 'data\mimic-iii-clinical-database-demo-1.4/PATIENTS.csv',
    'services': 'data\mimic-iii-clinical-database-demo-1.4/SERVICES.csv'}


class DataFrameLoader:
    def __init__(self):
        """
        Initialize the DataFrameLoader with a dictionary of file paths.

        :param file_paths: A dictionary where keys are identifiers for the
                           dataframes and values are the file paths for the CSV files.
        """
        self.file_paths = file_paths

    def load_dataframes(self):
        """
        Load CSV files into pandas DataFrames.

        :return: A dictionary of pandas DataFrames. Keys match the keys provided
                 in the file_paths dictionary.
        """
        dataframes = {}
        print("Loading into pandas DataFrames...")
        for key, path in self.file_paths.items():
            try:
                dataframes[key] = pd.read_csv(path)
                print(f"  {key} loaded successfully.")
            except FileNotFoundError as e:
                print(f"  Error loading {key}: {e}")
            except pd.errors.ParserError as e:
                print(f"  Error parsing {key}: {e}")
            except Exception as e:
                print(f"  Unexpected error loading {key}: {e}")
        return dataframes
