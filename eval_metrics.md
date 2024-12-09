# Evaluation Report
## ImmoEliza KNN Regression Model

### Content

* Code snippet of the model instantiation and parameters,...
* Evaluation metrics: 
  * MAE (on training/test)
  * RMSE (on training/test)
  * R2 (on training/test)
  * MAPE (on training/test)
  * sMAPE (on training/test)
  
* List of features and how you got it (to quickly understand if you've done data leakage)
  
* Accuracy computing procedure (on a test set? What split %, 80/20, 90/10, 50/50? k-fold cross?)

* Efficiency (training and inference time). 

* Quick presentation of the final dataset 
  * How many records, 
  * did you merge some datasets together? 
  * did you scrape again? 
  * what cleaning step you've done, scaling, encoding, cross-validation. (No need of visuals, just bullet points)

### Model Instantiation Code

The mode is instantiated using a `KNeighborsRegressor` from the `scikit-learn` library. Key hyperparameters and configurations include:

* **Number of Neighbors (n_neighbors):** Set to 8, based on model tuning for optimal performance.
  
* **Weights:** Distance-based weighting (`weights`) ensures predictions are influenced more by closer neighbors.
  
* **Cross-Validation:** A 5-fold cross-validation (`n_splits`) is implemented, which splits the dataset into five equal parts to rotate training and validation roles.
  
* **Scaling:** Features are standardized using a scaler to ensure that the KNN algorithm, which is sensitive to feature magnitude, works optimally. The dataset was scaled after splitting of test- and training-set to avoid data leakage.

Below is an extract of the key code for this process:

* **main.py**
    ```python
    # Assign target and feature variables (convert to NumPy array for compatibility with k-fold)
    y = df[TARGET].values
    X = df[FEATURES].values

    # Train model & perform k-fold cross-validation
    results = ModelUtils.train_and_cross_validate(
        X, y, n_splits=CV_N_SPLITS, n_neighbors=KNN_N_NEIGHBORS
    )
    ```

* **config.py**
    ```python
    KNN_N_NEIGHBORS = 8

    CV_N_SPLITS = 5

    TARGET = "price"

    FEATURES = [
        "living_area",
        "com_avg_income",
        "building_condition_encoded",
        "subtype_of_property_encoded",
        "latitude",
        "longitude",
        "equipped_kitchen_encoded",
        "min_distance",
        "terrace_encoded",
    ]
    ```

* **model_utils.py**
    ```python
    from sklearn.neighbors import KNeighorsRegressor
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler

    class ModelUtils:
        _scaler: StandardScaler = StandardScaler()
        _model: KNeighborsRegressor = None

    [...]

    @classmethod
    def train_and_cross_validate(
        cls, X: np.ndarray, y: np.ndarray, n_splits: int = 5, n_neighbors: int = 5
    ) -> Dict[str, Any]:
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
        [...]

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            # Standardize the data
            X_train_scaled = cls._scaler.fit_transform(X_train)
            X_test_scaled = cls._scaler.transform(X_test)

            # Train Model & Measure training time
            start_train_time = time.time()
            cls._model = KNeighborsRegressor(
                n_neighbors=n_neighbors, weights="distance"
            )
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

            [...]
    ```

The full implementation can be found in *main.py* and *model_utils.py* and includes timing for training and inference, metric calculations, and cross-validation.

**Key Parameters:**

* `n_splits=5`: Creates 5 subsets of data.
* `shuffle=True`: Randomly shuffles the data before splitting, preventing any ordering biases.
* `random_state=42`: Ensures reproducibility. This fixed seed guarantees that the shuffling and splits remain the same every time the code is run.

**Scaling:**

* KNN measure distances between points to determine neighbors. Features with larger magnitude can dominate the distance calculation and skew predictions.
* Scaling standardizes features to a standard normal distribution (mean = 0, variance = 1),  ensuring all features contribute equally.
* For each fold in the k-fold cross-validation (`kf.split(X)`), the training and test sets are split first and then scaled to ensure that no information from the test set is used when calculating the scaling parameters.


### Model Evaluation Metrics

**Cross-Validation Metrics (Averaged Across Folds) for Best Model:**

* Train Set Metrics:
  * MAE: 4137.78
  * RMSE: 14208.59
  * R²: 0.9904
  * MAPE: 1.14%
  * sMAPE: 1.12%

* Test Set Metrics:
  * MAE: 51453.95
  * RMSE: 79155.82
  * R²: 0.7022
  * MAPE: 16.38%
  * sMAPE: 14.79%

These metrics suggest the model performs well on the training data but shows reduced performance on the test data, which might warrant further feature refinement to improve generalization.

### Features Used

The final set of features and their sources:

* `living_area`: The property's living area in square meters as scraped from Immoweb.be.

* `building_condition_encoded`: The property's building condition. Scraped from Immoweb. Label-encoded based on quality.

* `subtype_of_property_encoded`: The property's subtype. Scraped from Immoweb, reduced to fice categories and label-encoded by quality.

* `equipped_kitchen_encoded`: The availability and kind of kitchen equipment. Scraped from Immoweb, label-encoded by quality.

* `terrace_encoded`: The availability of a terrace or not. Scraped from Immoweb. Binary feature (1 if terrace present, 0 otherwise).

* `com_avg_income`: Average taxable income per Belgian commune, sourced from Statbel, the Belgian statistical office (https://statbel.fgov.be/en/themes/households/taxable-income#figures).

* `latitude` and `longitude`: Obtained using Nominatim (geopy) and merged with real estate data.

* `min_distance`: Calculated as the Harvesine distance to the nearest of Belgium's 10 largest cities.


**Feature Engineering and Selection:**

* Features were selected based on:
  * Correlation coefficients (Spearman method).
  * Contribution to model metrics (evaluated iteratively).

* Strongly correlated features (e.g. `bedroom_nr`) were excluded to avoid redundancy and potential data leakage.

* Encoding:
  * Categories like kitchen equipment, property subtype, and building condition were label-encoded to represent ordinal relationships.
  * Binary encoding was used for terrace availability, due to inconsistencies in the availability of the indication of terrace surface on Immoweb.

* Features excluded despite better evaluation metrics to avoid overweighing location data:
  * `zip_code` and `refnis_code` (reformed NIS-code for communes)

### Accuracy Computing Procedure

**Validation Approach:**

* A 5-fold cross-validation was used:

  * Splits the dataset into 5 subsets (folds), using one for testing and the remaining four for training in each iteration.

  * Each fold serves as a validation set once, while the remaining folds are used for training.
  
* Cross-validation ensures the model is robust and is evaluated multiple times on different subsets of the data, which prevents the model from being overly reliant on any one test set. By training on different parts of the data, it ensures that the model learns from the data as a whole, rather than memorizing a specific split.
  
* Cross-validation is particularly important for KNN models because they are non-parametric and rely heavily on local patterns in the data. This approach evaluates the model's ability to generalize to unseen data more reliably.

**Train-Test-Split:**

* The dataset was split into training and validation sets within each fold.

* Results are averaged across the 5 folds to provide consistent metrics.

### Efficiency

* **Training Time:** 0.1343 seconds
* **Infrerence Time:** 4.3527 seconds

### Final Dataset Overview

**Records:**

* The final dataset includes 22.314 cleaned and preprocessed records.

**Data Sources:**

* Scraped data from Immoweb.be (property details).

* Statbel datasets for income and commune NIS-codes
  * https://statbel.fgov.be/en/themes/households/taxable-income#figures
  * https://statbel.fgov.be/nl/over-statbel/methodologie/classificaties/geografie

* Geopy-derived latitude and longitude data.

**Preprocessing Steps:**

* Cleaning: Removed outliers and imputed missing values based on mode or median per category.

* Encoding: Ordinal and binary encoding for categorical variables.

* Scaling: Standardized numerical features to improve KNN performance.
  
* Feature Selection: Eliminated redundant and highly correlated features based on statistical analysis.

* Cross-Validation: Ensured robust evaluation with k-fold splits.


### Additional Notes

* **Distance-Based Analysis:** The average neighbor distances for the KNN regression model are calculated and plotted. This helps in assessing how close the neighbors are on average for each data point, which can indicate how well the model is capturing the underlying data distribution. Large distances can suggest a model overfitting or underfitting, depending on whether the neighbors are too distant.

* **Permutation Importance:** The importance of each feature in predicting the target variable (price) is evaluated using permutation importance. This method helps quantify how much each feature contributes to the model's accuracy by randomly shuffling feature values and observing the effect on the model's performance.

* **SHAP:** SHAP (Shapley Additive Explanations) was considered for model interpretability but was abandoned due to its high computational cost when combined with k-fold cross-validation. The execution time for SHAP exceeded 30 minutes, making it impractical for this specific use case.

* **Sample Predictions and Plotting:** The model generates 10 sample predictions for a random subset of the test data. These predictions are compared with the actual target values and visualized in a plot, which allows for a quick inspection of the model’s performance in terms of real-world application.

The graphs resulting from the plotting of the results of the distance-based analysis and permutation importance are stored in the folder "./graphs".