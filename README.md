# ImmoEliza Project

**Part 3: KNN Regression Model**

## Project Description

This project is part of the ImmoEliza challenge, aimed at predicting real estate prices in Belgium using a K-Nearest Neighbors (KNN) regression model. The dataset, scraped and analyzed in previous steps, was cleaned, engineered, and formatted to produce a machine learning model that outputs predictions based on critical features such as location, property type, and amenities.

The goal of this project is to accurately predict property prices while demonstrating robust data preprocessing and insightful feature engineering.

## Features Implemented

### **Data Cleaning**

* Handled missing values:
    * Dropped rows with missing target values (`price`).
    * Imputed missing `facade_number` values using the median grouped by `subtype_of_property`.
    * Imputed `building_condition` using the mode.

* Removed unrealistic outliers:
  * Removed outliers in `price` based on the IQR.
  * Removed manually unrealistic outliers in `living_area`, `facade_number`, and `bedroom_nr`. These features were cleaned based on reasonable thresholds.
  
* Simplified categories:
  * Reduced property subtypes and encoded them for ML readiness.

### **Feature Engineering**

* Added external data:
  * Integrated NIS codes and taxable income per commune based on datasets by Statbel.fgov.be:
    * https://statbel.fgov.be/nl/over-statbel/methodologie/classificaties/geografie
    * https://statbel.fgov.be/en/themes/households/taxable-income#figures

  * Integrated latitude and longitude imported using Nominatim (geopy).

* Encoded categorical data:

    * Encoded categorical data using ordinal and manual mappings.

    * Transformed `terrace` to binary, due large amount of missing values for the terrace surface on Immoweb.

* Calculated additional features:

    * Calculated as the Harvesine distance to the nearest of Belgium's 10 largest cities based on the imported latitude and longitude.

### **Model Implementation**

* Trained a KNN regression model using k-fold cross-validation to ensure robust performance.

* Evaluated model performance through metrics like Mean Absolute Error (MAE) and R² score.

* Visualized results with:
  * True vs. predicted values.
  * Average neighbor distances.
  * Permutation importance.

## Directory Structure

```plaintext
challenge-regression/
├── main.py                             # Main script to run the project
├── config.py                           # Configuration file for parameters and file paths
├── coordinates.py                      # Script to fetch and save location data (coordinates)
├── data_utils.py                       # Utility classes for data cleaning and preprocessing
├── feature_utils.py                    # Utility classes for feature engineering and dataset preparation
├── model_utils.py                      # Utility classes for model fitting, training, and evaluation
├── requirements.txt                    # Required Python libraries
├── data/                               # Folder containing datasets
│   ├── real_estate.csv                 # Original dataset
│   ├── real_estate_w_coordinates.csv   # Dataset with added latitude and longitude
├── graphs/                             # Folder for saving performance graphs
│   ├── predictions.png                 # Predictions vs true values graph
│   ├── distances.png                   # Distance analysis graph
│   ├── permutation_importance.png      # Permutation importance graph
└── README.md                           # This file
```

## Installation

### Prerequisites

Ensure you have the following software installed:
* Python 3.7+
* pip (Python package isntaller)
* git (for cloning the repository)

### Steps

1. Clone this repository and go to directory:

    ```bash
    git clone https://github.com/M-0612/challenge-regression.git
    cd challenge-regression
    ```

2. Install required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Ensure the following data files are available:
  * `data/real_estate.csv`: Original dataset. Used to fetch latitude and longitude for each commune.
  * `data/real_estate_w_coordinates.csv`: Dataset with latitude and longitude (already available, no need to rerun `coordinates.py`)    


## Usage

1. Edit the configuration in `config.py` if required to specify file paths and parameters.

2. Run the main script:
    ```bash
    python main.py
    ```
3. Outputs:
  * Evaluation Metrics:
    * Mean Absolute Error (MAE), R² score, and other relevant metrics are printed to the console.

  * Performance Graphs:
    * Saved in the `/graphs` folder as:
      * `predicitons.png`: Scatterplot of predictions vs true values.
      * `avg_distances.png`: Histogram of average neighbor distances.
      * `permutation_importance.png`: Bar graph of permutation feature importance.

## File Descriptions

* `main.py`: Contains the main script to run the data pipeline, train the model, and generate performance outputs. Uses the dataset augmented with latitude and longitude columns `real_estate_w_coordinates.csv`.

* `data_utils.py`: Provides a utility class and methods for data cleaning and preprocessing tasks such as handling missing values, dropping outliers, and merging external datasets.

* `feature_utils.py`: Includes methods for feature engineering, such as encoding categorical variables, creating new features, and calculating e.g. distances to large cities.

* `model_utils.py`: Contains methods for splitting and standardizing the dataset, fitting the KNN model, evaluating its performance, and visualizing results.

* `coordinates.py`: 
  * Fetches longitude and latitude for location-based features (e.g. communes) and implements the coordinates in the given DataFrame.
  * Generates:
    * `real_estate_w_coordinates.csv`: Dataset including the fetched latitude and longitude values, used in `main.py`.
    * A dictionary of city names and their coordinates for feature engineering (calculation of distance to nearest city).
  * **Note:** This script does not need to be rerun, as coordinates remain static, and the augmented dataset used in `main.py`is already provided in the `/data` folder.


* `config.py`: Contains parameters and file paths that can be adjusted if required.


## Contributors

* Miriam Stoehr

## Personal Situation

This KNN regression model was developed as a capstone project for the BeCode Data Science and AI Bootcamp 2024/2025. The project provided an opportunity to gain hands-on experience in key aspects of the machine learning workflow, including data preprocessing, feature engineering, and model evaluation.

Primary focus during this project was to:

* Handle missing values and outliers effectively to ensure a clean and reliable dataset.
* Encode categorical data to make it compatible with machine learning models.
* Integrate external datasets, such as income and city coordinates, to enrich the dataset and provide meaningful context for predictions.
* Perform relevant feature engineering to improve the model’s performance without introducing data leakage, such as computation of new features and reducing unnecessary complexity in categories.



