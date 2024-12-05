import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

class FeatureUtils:
    @classmethod
    def calc_avg(cls, df: pd.DataFrame, new_col: str, dividend: str, divisor: str) -> pd.DataFrame:
        df[new_col] = df[dividend] / df[divisor]
        return df
    
    @classmethod
    def encode_by_group_avg(cls, df: pd.DataFrame, group_col: str, avg_col: str) -> pd.DataFrame:
        # Calc average of avg_col for each group of group_col
        group_avg = df.groupby(group_col)[avg_col].mean()
        # Sort group values by average of avg_col by descending order
        sorted_group = group_avg.sort_values(ascending=False)
        # Reverse enumerate so that highest value gets highest number
        group_label_mapping = {g: label for label, g in enumerate(sorted_group.index[::-1], start=1)}
        # Apply mapping
        df[ f'{group_col}_encoded'] = df[group_col].map(group_label_mapping)
        return df
    
    @classmethod
    def encode_binary(cls, df: pd.DataFrame, column: str, threshold: float = 0) -> pd.DataFrame:
        """
        Converts a specified column to binary values based on a threshold.

        Args:
            df (pd.DataFrame): The DataFrame to be processed.
            column (str): The column to binarize.
            threshold (float): The threshold value to determine binary classification.

        Returns:
            pd.DataFrame: The DataFrame with the specified column converted to binary values.
        """
        df[f'{column}_encoded'] = (df[column] > threshold).astype(int)
        return df
    
    @classmethod
    def map_manual(cls, df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
        """
        Reduces categorical values in the same column based on a given mapping.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): The column to reduce categorical values.
            mapping (dict): A dictionary that maps old values to new ones.

        Returns:
            pd.DataFrame: DataFrame with reduced categories in the given column.
        """
        df.loc[:, column] = df.loc[:, column].map(mapping)
        return df
    
    @classmethod
    def encode_manual(cls, df: pd.DataFrame, column: str, mapping: dict) -> pd.DataFrame:
        """
        Replaces categorical values in a separate column based on a given mapping.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): The column to reduce categorical values.
            mapping (dict): A dictionary that maps old values to new ones.

        Returns:
            pd.DataFrame: DataFrame with reduced categories in the given column.
        """
        df[f'{column}_encoded'] = df[column].map(mapping)
        return df


    @classmethod
    def encode_ordinal(cls, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Performs ordinal encoding of categorical values.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): The column whose values should be encoded.

        Returns:
            pd.DataFrame: DataFrame with encoded values.
        """
        # Initialize OrdinalEncoder
        encoder = OrdinalEncoder()
        # Fit and transform the column
        df[f'{column}_encoded'] = encoder.fit_transform(df[[column]])
        return df
    
    @classmethod
    def convert_dtype(cls, df: pd.DataFrame, column: str, conv_type: str) -> pd.DataFrame:
        """
        Converts the data type of a column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column name to process.
            conv_type (str): Target data type (e.g., 'int', 'float').

        Returns:
            pd.DataFrame: DataFrame with the updated column type.
        """
        df[column] = df[column].astype(conv_type)
        return df

    @classmethod
    def select_features(cls, df: pd.DataFrame, features: list) -> pd.DataFrame:
        """
        Produces a subset based on the provided column names.

        Args:
            df (pd.DataFrame): Input DataFrame.
            features (list): List of the columns to be selected.

        Returns:
            pd.DataFrame: Resulting subset.
        """
        df_selected = df[features]
        return df_selected
