import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Union, Dict
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class ModelUtils:
    """
    A utility class for handling KNN regression models, including data standardization,
    training, prediction, and evaluation.
    """
    _scaler: StandardScaler = StandardScaler()
    _model: KNeighborsRegressor = None

    @classmethod
    def standardize_data(cls, data: Union[pd.DataFrame, np.ndarray], set_type: str = 'train') -> np.ndarray:
        """
        Standardize data using StandardScaler to avoid data leakage.

        Args:
            data (Union[pd.DataFrame, np.ndarray]): The data to standardize.
            set_type (str): Either 'train' to fit and transform, or 'test' to transform only.

        Returns:
            np.ndarray: Standardized data.
        """
        if set_type == 'train':
            return cls._scaler.fit_transform(data)
        elif set_type == 'test':
            return cls._scaler.transform(data)
        else:
            raise ValueError("set_type must be 'train' or 'test'")

    @classmethod
    def train_model(cls, X_train: np.ndarray, y_train: np.ndarray, n_neighbors: int = 5) -> None:
        """
        Train the KNeighborsRegressor model.

        Args:
            X_train (np.ndarray): Scaled training features.
            y_train (np.ndarray): Training labels.
            n_neighbors (int): Number of neighbors to use for KNN. Defaults to 5.
        """
        cls._model = KNeighborsRegressor(n_neighbors=n_neighbors)
        cls._model.fit(X_train, y_train)

    @classmethod
    def predict(cls, X_test: np.ndarray) -> np.ndarray:
        """
        Predict using the trained KNeighborsRegressor model.

        Args:
            X_test (np.ndarray): Scaled test features.

        Returns:
            np.ndarray: Predicted values.
        """
        if cls._model is None:
            raise ValueError("Model has not been trained yet.")
        return cls._model.predict(X_test)

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model using R² and MAE.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            Dict[str, float]: A dictionary containing R² and MAE score.
        """
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)

        return {
            "Mean Absolute Error (MAE)": mae,
            "Mean Squared Error (MSE)": mse,
            "Root Mean Squared Error (RMSE)": rmse,
            "R Squared": r2
        }
    
    @staticmethod
    def plot_predictions(y_true: np.ndarray, y_pred: np.ndarray, title: str = 'Predictions vs True Values') -> None:
        """
        Plot predictions against true values.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            title (str): Title of the plot.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(y_true, y_pred, alpha=0.6, color='blue', edgecolor='k', label='Predictions')
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', linewidth=2, label='Ideal Line (y=x)')
        plt.title(title)
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.legend()
        plt.grid(True)
        plt.show()
    
    @staticmethod
    def print_sample_predictions(y_true: np.ndarray, y_pred: np.ndarray, num_samples: int = 10) -> None:
        """
        Print a sample of predictions vs true values.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            num_samples (int): Number of samples to display.
        """
        sample_df = pd.DataFrame({
            "True Value": y_true,
            "Predicted Value": y_pred
        }).head(num_samples)

        print("\nSample Predictions:")
        print(sample_df.to_string(index=False, float_format="{:.2f}".format))