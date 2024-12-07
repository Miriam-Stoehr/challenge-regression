import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List
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
    
    @staticmethod
    def calculate_avg_neighbor_distances(X_test_scaled: np.ndarray) -> np.ndarray:
        """
        Calculate the average distances of test points to their k-nearest neighbors.

        Args:
            X_test_scaled (np.ndarray): Scaled test features.

        Returns:
            np.ndarray: Average distances for each test instance.
        """
        distances, _ = ModelUtils._model.kneighbors(X_test_scaled)
        avg_distances = distances.mean(axis=1)
        return avg_distances
    
    @staticmethod
    def plot_avg_neighbor_distances(avg_distances: List[float]) -> None:
        """
        Visualize the distribution of average neighbor distances.

        Args:
            avg_distances (List[float]): A list of average distances to nearest neighbors for test instances.
        
        Returns:
            None: This function generates a histogram plot with a KDE (Kerne Density Estimate).
        """
        plt.figure(figsize=(10, 6))
        sns.histplot(avg_distances, kde=True, bins=30, color='skyblue')
        plt.title("Distribution of Average Neighbor Distances", fontsize=16)
        plt.xlabel("Average Distances", fontsize=14)
        plt.ylabel("Frequency", fontsize=14)
        plt.grid(alpha=0.4)
        plt.show()

    @staticmethod
    def permutation_importance(X: np.ndarray, y: np.ndarray, n_repeats: int = 10) -> Dict[str, float]:
        """
        Compute permutation importance for features.

        Args:
            X (np.ndarray): Feature set.
            y (np.ndarray): Target values.
            n_repeats (int): Number of shuffling repetitions.
        
        Returns:
            Dict[str, float]: Importance scores for each feature.
        """
        baseline_score = r2_score(y, ModelUtils._model.predict(X))
        importances = []

        for i in range(X.shape[1]):
            shuffled_scores = []
            for _ in range(n_repeats):
                X_shuffled = X.copy()
                np.random.shuffle(X_shuffled[:, i])  # Shuffle column i
                score = r2_score(y, ModelUtils._model.predict(X_shuffled))
                shuffled_scores.append(baseline_score - score)
            
            avg_importance = np.mean(shuffled_scores)
            importances.append(avg_importance)
        
        return dict(zip(range(X.shape[1]), importances))
    
    @staticmethod
    def plot_permutation_importance(permutation_scores: Dict[int, float], feature_names: List[str]) -> None:
        """
        Visualize feature importance based on permutation scores.
        
        Args:
            permutation_scores (Dict[int, float]): A dictionary where the keys are feature indices 
                                                and the values are the computed importance scores.
            feature_names (List[str]): A list of feature names corresponding to the feature indices.
        
        Returns:
            None: This function generates a horizontal bar chart to display feature importance.
        
        Example:
            feature_names = ['price', 'latitude', 'longitude', 'living_area', 'garden', 
                            'subtype_of_property', 'building_condition', 'equipped_kitchen', 
                            'terrace', 'swimming_pool', 'facade_number']
            
            permutation_scores = ModelUtils.permutation_importance(X_scaled, y, n_repeats=10)
            plot_permutation_importance(permutation_scores, feature_names)
        """
        # Sort permutation scores by importance (descending order)
        sorted_importances = sorted(permutation_scores.items(), key=lambda x: x[1], reverse=True)
        features, importances = zip(*sorted_importances)
        feature_labels = [feature_names[i] for i in features]
        
        # Plot horizontal bar chart
        plt.figure(figsize=(12, 6))
        plt.barh(feature_labels, importances, color='teal')
        plt.title("Permutation Feature Importance", fontsize=16)
        plt.xlabel("Importance Score", fontsize=14)
        plt.ylabel("Features", fontsize=14)
        plt.gca().invert_yaxis()  # Invert y-axis for better visualization
        plt.grid(alpha=0.4)
        plt.show()