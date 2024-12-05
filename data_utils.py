import pandas as pd
from typing import Optional

class DataUtils:
    """
    A utility class for handling common data preprocessing tasks.
    """

    @staticmethod
    def import_csv(file_path: str) -> pd.DataFrame:
        """
        Imports a CSV file into a DataFrame.

        Args:
            file_path (str): Path to the CSV file.

        Returns:
            pd.DataFrame: The imported data.
        """
        return pd.read_csv(file_path)

    @staticmethod
    def export_csv(df: pd.DataFrame, file_path: str) -> None:
        """
        Exports a DataFrame to a CSV file.

        Args:
            df (pd.DataFrame): Data to be exported.
            file_path (str): Path where the CSV file will be saved.
        """
        df.to_csv(file_path, index=False)

    @classmethod
    def handle_missing_values(cls, df: pd.DataFrame, column: str, strategy: str = 'fill', fill_value: str = 'Unknown') -> pd.DataFrame:
        """
        Handles missing values in a specified column.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column name to process.
            strategy (str): Strategy to handle missing values ('fill' or 'drop').
            fill_value (str): Value to fill if strategy is 'fill'.

        Returns:
            pd.DataFrame: DataFrame with missing values handled.

        Raises:
            ValueError: If an invalid strategy is provided.
        """
        if strategy == 'fill':
            df.loc[:, column] = df[column].fillna(fill_value)
        elif strategy == 'drop':
            df.dropna(subset=[column], inplace=True)
        else:
            raise ValueError("Strategy must be 'fill' or 'drop'.")
        return df

    @classmethod
    def fill_missing_with_mode(
        cls, df: pd.DataFrame, column: str, strategy: str = 'fill'
    ) -> pd.DataFrame:
        """
        Handles missing values for a specific column by filling them with the mode.

        Args:
            df (pd.DataFrame): The DataFrame to be processed.
            column (str): The column name to handle missing values.
            strategy (str): The strategy to handle missing values ('fill' or 'drop').

        Returns:
            pd.DataFrame: The DataFrame with missing values filled with the mode or dropped.
        """
        if strategy == 'fill':
            mode_value = df[column].mode()[0]
            df.loc[:, column] = df[column].fillna(mode_value)
        elif strategy == 'drop':
            df.dropna(subset=[column], inplace=True)
        else:
            raise ValueError("Strategy must be 'fill' or 'drop'.")
        return df

    @classmethod
    def fill_missing_by_group(
        cls, df: pd.DataFrame, column: str, group_column: str, agg_func: str = 'median'
    ) -> pd.DataFrame:
        """
        Fills missing values in a column based on a group-specific statistic (e.g., median, mean).

        Args:
            df (pd.DataFrame): The DataFrame to be processed.
            column (str): The column with missing values to be filled.
            group_column (str): The column to group by (e.g., 'subtype_of_property').
            agg_func (str): The aggregation function to use ('median', 'mean').

        Returns:
            pd.DataFrame: The DataFrame with missing values filled based on the group-specific statistic.
        """
        if agg_func == 'median':
            fill_values = df.groupby(group_column)[column].median()
        elif agg_func == 'mean':
            fill_values = df.groupby(group_column)[column].mean()
        else:
            raise ValueError("Aggregation function must be 'median' or 'mean'.")
        
        df.loc[:, column] = df.apply(
            lambda row: fill_values[row[group_column]] if pd.isnull(row[column]) else row[column],
            axis=1
        )
        return df

    @classmethod
    def remove_substring(cls, df: pd.DataFrame, column: str, substring: str) -> pd.DataFrame:
        """
        Removes a specified substring from all values in a given column.

        Args:
            df (pd.DataFrame): The DataFrame to be processed.
            column (str): The column from which to remove the substring.
            substring (str): The substring to remove.

        Returns:
            pd.DataFrame: The DataFrame with the substring removed from the specified column.
        """
        df.loc[:, column] = df[column].str.replace(substring, "", regex=False)
        return df

    @classmethod
    def correct_value(cls, df: pd.DataFrame, column: str, old_value: str|int, new_value: str|int) -> pd.DataFrame:
        df.loc[df[column] == old_value, column] = new_value
        return df

    @classmethod
    def drop_outliers(cls, df: pd.DataFrame, column: str, lower_bound: Optional[float] = None, upper_bound: Optional[float] = None) -> pd.DataFrame:
        """
        Drops rows with values in a column outside specified bounds.

        Args:
            df (pd.DataFrame): Input DataFrame.
            column (str): Column name to filter.
            lower_bound (Optional[float]): Minimum valid value.
            upper_bound (Optional[float]): Maximum valid value.

        Returns:
            pd.DataFrame: DataFrame without outliers.
        """
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = lower_bound if lower_bound is not None else Q1 - 1.5 * IQR
        upper_bound = upper_bound if upper_bound is not None else Q3 + 1.5 * IQR

        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
        return df
