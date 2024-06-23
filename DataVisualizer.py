import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualizer:
    def __init__(self, dataframe):
        """
        Initialize the DataVisualizer with the dataframe to be visualized.

        :param dataframe: A pandas DataFrame to be used for visualizations.
        """
        self.dataframe = dataframe

    def plot_missing_data(self):
        """
        Plots the percentage of missing data for each column in the dataframe.
        """
        missing_data_ratios = (
            self.dataframe.isnull().sum() / len(self.dataframe)) * 100
        missing_data_ratios = missing_data_ratios[missing_data_ratios > 0].sort_values(
            ascending=False)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=missing_data_ratios.index, y=missing_data_ratios)
        plt.ylabel('Percentage of Missing Data')
        plt.xlabel('Columns')
        plt.title('Percentage of Missing Data by Column')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def plot_patient_outcomes_by_admission_type(self):
        """
        Plots patient outcomes by admission type.
        """
        plt.figure(figsize=(10, 6))
        sns.countplot(
            x='admission_type',
            hue='hospital_expire_flag',
            data=self.dataframe,
            palette='Set1')
        plt.title('Patient Outcomes by Admission Type')
        plt.xlabel('Admission Type')
        plt.ylabel('Count')
        plt.legend(
            title='Hospital Expire Flag',
            labels=[
                'Survived',
                'Deceased'])
        plt.tight_layout()
        plt.show()

    def plot_average_los_by_admission_and_insurance(self):
        """
        Plots the average length of stay by admission type and insurance.
        """
        grouped_data = self.dataframe.groupby(['admission_type', 'insurance'])[
            'los'].mean().reset_index()

        plt.figure(figsize=(10, 6))
        sns.barplot(
            x='admission_type',
            y='los',
            hue='insurance',
            data=grouped_data,
            palette='Set2')
        plt.title('Average Length of Stay by Admission Type and Insurance')
        plt.xlabel('Admission Type')
        plt.ylabel('Average Length of Stay (days)')
        plt.legend(title='Insurance Type')
        plt.tight_layout()
        plt.show()
