import pandas as pd

class DataFrameLoader:
    def __init__(self, file_paths):
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
        for key, path in self.file_paths.items():
            try:
                dataframes[key] = pd.read_csv(path)
                print(f"{key} loaded successfully.")
            except FileNotFoundError as e:
                print(f"Error loading {key}: {e}")
            except pd.errors.ParserError as e:
                print(f"Error parsing {key}: {e}")
            except Exception as e:
                print(f"Unexpected error loading {key}: {e}")
        return dataframes
