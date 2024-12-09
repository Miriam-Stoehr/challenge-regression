import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from typing import Dict, List

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
    def encode_ordinal_by_value(cls, df: pd.DataFrame, group_column: str, avg_column: str, new_column: str) -> pd.DataFrame:
        """
        Calculate the average 'com_avg_income' per 'commune' and apply labels based on the average income.
        The highest label corresponds to the highest average income.
        
        Args:
            df (pd.DataFrame): The input DataFrame.
            group_column (str): The column name containing the data based on which to group the values.
            avg_column (str): The column name containing the data based on which to calculate the average.
            new_column (str): The name of the new column.

        
        Returns:
            pd.DataFrame: DataFrame with an additional label column.
        """
        # Calculate the average income per commune
        avg_per_group = df.groupby(group_column)[avg_column].mean().reset_index()
        
        # Sort communes by the average income in descending order
        avg_per_group = avg_per_group.sort_values(by=avg_column, ascending=True)
        
        # Create a mapping of commune to label (highest average income gets the highest label)
        group_to_label = {group: label for label, group in enumerate(avg_per_group[group_column])}
        
        # Map the labels back to the original DataFrame
        df[new_column] = df[group_column].map(group_to_label)
        
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

    @staticmethod
    def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the Haversine distance between two points on the Earth specified by their latitude and longitude.

        Args:
            lat1 (float): First latitude point.
            lon1 (float): First longitude point.
            lat2 (float): Second latitude point.
            lon2 (float): Second longitude point.
        
        Returns:
            float: Haversine distance between two sets of coordinates, measured in kilometers.
        """
        # Convert degrees to radians
        lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = sin(dlat / 2)**2 + cos(lat1) * cos(lat2) * sin(dlon / 2)**2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        R = 6371 # Radius of Earth in km
        return R * c

    @classmethod
    def calculate_nearest_city_distance(cls, df: pd.DataFrame, cities: Dict[str, List[float]]) -> pd.DataFrame:
        """
        Calculates the minimum distance from each commune to the nearest city and adds it as a new column.
        
        Args:
            df (pd.DataFrame): DataFrame with communes and their latitude/longitude columns.
            cities (Dict[str, List[float]]): Dictionary with city names as keys and their (latitude, longitude) as values.
        
        Returns:
            pd.DataFrame: The DataFrame with an additional 'min_distance' column.
        """
        # Calculate the minimum distance to the nearest city for each commune
        min_distances = []

        for _, row in df.iterrows():
            lat_commune = row['latitude']
            lon_commune = row['longitude']

            min_distance = float('inf')   # Start with an infinitely large distance

            # Iterate through each city in the dictionary and calculate the distance
            for city, (lat_city, lon_city) in cities.items():
                distance = cls.haversine(lat_commune, lon_commune, lat_city, lon_city)
                if distance < min_distance:
                    min_distance = distance
            
            min_distances.append(min_distance)
        
        # Add the minimum distances to the DataFrame as a new column
        df['min_distance'] = min_distances
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

    