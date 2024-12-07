import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from typing import List

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
    def merge_data(cls, curr_df: pd.DataFrame, ext_df: pd.DataFrame, import_col: List[str], join_left: str, join_right: str) -> pd.DataFrame:
        """
        Merges columns from an external dataframe into the current dataframe and drops external columns on which the merge is performed.

        Args:
            curr_df (pd.DataFrame): Current DataFrame.
            ext_df (pd.DataFrame): External DataFrame.
            import_col (List[str]): List of columns to import.
            join_left (str): Column of the current df based on which the merge should be performed.
            join_right (str): Column of the external df based on which the merge should be performed.
        
        Returns:
            pd.DataFrame: Updated DataFrame.
        """
        # Merge colums from ext_df in curr_df based on stated column
        updated_df = pd.merge(curr_df, ext_df[import_col], left_on=join_left, right_on=join_right, how='left')
        # Drop external column on which the merge happened
        updated_df = updated_df.drop(columns=[join_right])
        return updated_df

    @classmethod
    def rename_columns(cls, df: pd.DataFrame, columns: dict) -> pd.DataFrame:
        """
        Renames specified columns according to a given dictionary.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (dict): Dictionary that maps old and new column names.
        
        Returns:
            pd.DataFrame: Updated DataFrame.
        """
        updated_df = df.rename(columns=columns)
        return updated_df

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
    
    @staticmethod
    def cluster_dbscan(df: pd.DataFrame, columns: List[str], new_column: str, eps: float, min_samples: int) -> pd.DataFrame:
        """
        Build clusters based on the given columns. Resulting clusters will be labeled in descending order based on the first column given, with highest label number corresponding to highest value.

        Args:
            df (pd.DataFrame): Input DataFrame.
            clolumns (List[str]): List of columns to be used for clustering.
            eps (float): Epsilon value for DBSCAN (max. distance for neighbors).
            min_samples (int): Minimum number of samples to form a cluster.
        
        Returns:
            pd.DataFrame: DataFrame including an additional column for the clusters.
        """
        # Normalize features
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[columns])

        #Apply DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        df[new_column] = dbscan.fit_predict(scaled_features)

        # Get the unique cluster labels and reverse them
        unique_labels = sorted(df[new_column].unique(), reverse=True)
        label_mapping = {old_label: new_label for old_label, new_label in zip(sorted(df[new_column].unique()), unique_labels)}

        # Map the reversed labels back to the original DataFrame
        df[f'{new_column}_reversed'] = df[new_column].map(label_mapping)

        # Drop column with original order
        df = df.drop(columns=[new_column])

        # Rename reversed column
        df = df.rename(columns={f'{new_column}_reversed': new_column})

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

    