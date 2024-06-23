class DataPreprocessor:
    def __init__(self, dataframes):
        """
        Initialize the DataPreprocessor with the dataframes to be preprocessed.

        :param dataframes: A dictionary of pandas dataframes to be processed.
        """
        self.dataframes = dataframes

    def preprocess(self):
        """
        Perform the preprocessing steps on the dataframes.

        :return: A processed pandas dataframe.
        """
        # Merge dataframes
        print("Merging DataFrames...")
        admission_df = self.dataframes['admissions']
        callout_df = self.dataframes['callout']
        drgcodes_df = self.dataframes['drgcodes']
        icustays_df = self.dataframes['icustays']
        patient_df = self.dataframes['patients']
        services_df = self.dataframes['services']

        patient_df = patient_df.merge(admission_df, on="subject_id", how="left", suffixes=('_patient', '_admission')) \
                       .merge(callout_df, on=["subject_id", "hadm_id"], how="left", suffixes=('_admission', '_callout')) \
                       .merge(icustays_df, on=["subject_id", "hadm_id"], how="left", suffixes=('_callout', '_icustays')) \
                       .merge(drgcodes_df, on=["subject_id", "hadm_id"], how="left", suffixes=('_icustays', '_drgcodes')) \
                       .merge(services_df, on=["subject_id", "hadm_id"], how="left", suffixes=('_drgcodes', '_services'))

        # Drop empty and unnecessary columns
        patient_df.dropna(how="all", axis="columns", inplace=True)
#        print(patient_df.columns.tolist())


        columns_to_drop = [
            "row_id_x",
            "row_id_y",
            "dod_hosp",
            "dod_ssn",
            "language",
            "religion",
            "marital_status",
            "edregtime",
            "edouttime",
            "diagnosis",
            "has_chartevents_data"
            "submit_wardid",
            "submit_careunit",
            "curr_wardid",
            "curr_careunit",
            "callout_wardid",
            "request_tele",
            "request_resp",
            "request_cdiff",
            "request_mrsa"]
        patient_df.drop(columns_to_drop, axis=1, inplace=True, errors='ignore')

        # Convert columns to lowercase
        patient_df = patient_df.apply(lambda x: x.astype(
            str).str.lower() if x.dtype == "object" else x)

        # Fill 'callout_service' and 'callout_outcome' with mode
        for column in ["callout_service", "callout_outcome"]:
            mode = patient_df[column].mode()[0]
            patient_df[column] = patient_df[column].fillna(mode)

        # Fill 'createtime' and 'outcometime' with 'admittime'
        patient_df["createtime"] = patient_df["createtime"].fillna(
            patient_df["admittime"])
        patient_df["outcometime"] = patient_df["outcometime"].fillna(
            patient_df["admittime"])

        # Fill 'deathtime' with 'dod' for patients not having hospital death
        patient_df["deathtime"] = patient_df["deathtime"].fillna(
            patient_df["dod"])

        return patient_df
