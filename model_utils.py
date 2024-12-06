import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Any, Dict
from sklearn.model_selection import KFold
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

    @staticmethod
    def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model predictions using multiple metrics.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            Dict[str, float]: A dictionary containing multiple evaluation metrics.
        """
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100 if np.all(y_true) else np.inf
        smape = (
            100 * np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))
            if np.all(y_true + y_pred)
            else np.inf
        )

        return {
            "Mean Absolute Error (MAE)": mae,
            "Root Mean Squared Error (RMSE)": rmse,
            "Mean Squared Error (MSE)": mse,
            "R Squared": r2,
            "Mean Absolute Percentage Error (MAPE)": mape,
            "Symmetric Mean Absolute Percentage Error (sMAPE)": smape
        }

    @classmethod
    def cross_validate(cls, X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_neighbors: int = 5) -> Dict[str, Any]:
        """
        Perform k-fold cross-validation and compute metrics for training and test sets.

        Args:
            X (np.ndarray): Feature set.
            y (np.ndarray): Target variable.
            n_splits (int): Number of folds for cross-validation. Defaults to 5.
            n_neighbors (int): Number of neighbors for KNN. Defaults to 5.

        Returns:
            Dict[str, Any]: Average metrics and collected predictions.
        """
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        metrics_list = {"train": [], "test": []}
        all_y_true = []  # Collect true values
        all_y_pred = []  # Collect predictions

        # Track training and inference time
        total_train_time = 0
        total_inference_time = 0

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Standardize the data
            X_train_scaled = cls._scaler.fit_transform(X_train)
            X_test_scaled = cls._scaler.transform(X_test)

            # Train Model & Measure training time
            start_train_time = time.time()
            cls._model = KNeighborsRegressor(n_neighbors=n_neighbors, weights='distance')
            cls._model.fit(X_train_scaled, y_train)
            end_train_time = time.time()
            train_time = end_train_time - start_train_time
            total_train_time += train_time

            # Predict on training and test sets & Measure inference time
            start_inference_time = time.time()
            y_train_pred = cls._model.predict(X_train_scaled)
            y_test_pred = cls._model.predict(X_test_scaled)
            end_inference_time = time.time()
            inference_time = end_inference_time - start_inference_time
            total_inference_time += inference_time
            
            # Collect predictions for the test set
            all_y_true.extend(y_test)
            all_y_pred.extend(y_test_pred)

            # Evaluate and store metrics for training and test sets
            train_metrics = cls.evaluate(y_train, y_train_pred)
            test_metrics = cls.evaluate(y_test, y_test_pred)

            metrics_list["train"].append(train_metrics)
            metrics_list["test"].append(test_metrics)

        # Average metrics across all folds
        averaged_metrics = {
            "train": {
                metric: np.mean([fold[metric] for fold in metrics_list["train"]])
                for metric in metrics_list["train"][0]
            },
            "test": {
                metric: np.mean([fold[metric] for fold in metrics_list["test"]])
                for metric in metrics_list["test"][0]
            },
        }

        # Return metrics and collected predictions
        return {
            "metrics": averaged_metrics,
            "y_true": np.array(all_y_true),
            "y_pred": np.array(all_y_pred),
            "train_time": total_train_time,
            "inference_time": total_inference_time,
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