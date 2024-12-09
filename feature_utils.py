import pandas as pd
from math import radians, sin, cos, sqrt, atan2
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.cluster import DBSCAN
from typing import Dict, List


class FeatureUtils:
    """
    A utility class for handling feature preprocessing tasks.
    """

    @classmethod
    def encode_binary(
        cls, df: pd.DataFrame, column: str, threshold: float = 0
    ) -> pd.DataFrame:
        """
        Converts a specified column to binary values based on a threshold.

        Args:
            df (pd.DataFrame): The DataFrame to be processed.
            column (str): The column to binarize.
            threshold (float): The threshold value to determine binary classification.

        Returns:
            pd.DataFrame: The DataFrame with the specified column converted to binary values.
        """
        df[f"{column}_encoded"] = (df[column] > threshold).astype(int)
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
    def encode_manual(
        cls, df: pd.DataFrame, column: str, mapping: dict
    ) -> pd.DataFrame:
        """
        Replaces categorical values in a separate column based on a given mapping.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): The column to reduce categorical values.
            mapping (dict): A dictionary that maps old values to new ones.

        Returns:
            pd.DataFrame: DataFrame with reduced categories in the given column.
        """
        df[f"{column}_encoded"] = df[column].map(mapping)
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
        df[f"{column}_encoded"] = encoder.fit_transform(df[[column]])
        return df

    @classmethod
    def merge_data(
        cls,
        curr_df: pd.DataFrame,
        ext_df: pd.DataFrame,
        import_col: List[str],
        join_left: str,
        join_right: str,
    ) -> pd.DataFrame:
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
        updated_df = pd.merge(
            curr_df,
            ext_df[import_col],
            left_on=join_left,
            right_on=join_right,
            how="left",
        )
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
        a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))
        R = 6371  # Radius of Earth in km
        return R * c

    @classmethod
    def calculate_nearest_city_distance(
        cls, df: pd.DataFrame, cities: Dict[str, List[float]]
    ) -> pd.DataFrame:
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
            lat_commune = row["latitude"]
            lon_commune = row["longitude"]

            min_distance = float("inf")  # Start with an infinitely large distance

            # Iterate through each city in the dictionary and calculate the distance
            for city, (lat_city, lon_city) in cities.items():
                distance = cls.haversine(lat_commune, lon_commune, lat_city, lon_city)
                if distance < min_distance:
                    min_distance = distance

            min_distances.append(min_distance)

        # Add the minimum distances to the DataFrame as a new column
        df["min_distance"] = min_distances
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
