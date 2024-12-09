# Presentation: KNN Regression Model for Real-Estate Price Prediction

## 1. Introduction

**1.1 Context and Objective:** 
The real-estate market thrives on accurate price predictions. For an online real-estate platform, the ability to predict property prices accurately enhances user trust, attracts more clients, and enables better data-driven decision-making.

The project goal was to develop a reliable model for predicting property prices based on features like location, property characteristics and relevant external data, such as income statistics.

**1.2 Why KNN Regression?**
For this task the K-Nearest Neighbors (KNN) regression model provides:

* **Simplicity:** KNN is a relatively straightforward algorithm that makes predictions based on the similarity between data points.
* **Flexibility:** It works well for non-linear relationships, making it suitable for complex real-estate datasets.
* **Interpretability:** KNN's predictions can be directly traced back to the closest data points, making the decision-making process transparent and easy to explain.
* **Adaptability.** It allows for distance-weighted predicitons, which can capture localized variations in pricing effectively.

## 2. Dataset Analysis

**2.1 Overview of the Dataset**
The dataset contains current real-estate data, including features such as:

* Property Characteristics: Living area, property subtype, equipped kitchen, building condition, etc.
* Location: Latitude and longitude of the property.
* Additional Features: Minimum distance to major cities, communal average income.

**2.2 Feature Engineering**

* Categorical Encoding: Property types and building conditions were encoded numerically to make them compatible with the algorithm.
* Derived Features: Distances to major cities were computed to capture regional effects on pricing.
* Standardized Features: To ensure fair comparison across dimensions, continuous features (e.g., living area, latitude, longitude) were standardized.

**2.3 Dataset Challenges and Limitations**

* Imbalanced Distribution: Certain property types or locations may dominate the dataset, potentially biasing the model.
* Incomplete Data: Missing or inconsistent entries in features like condition or price could affect the model’s accuracy.
* Dynamic Market Trends: Real-estate prices fluctuate over time, making historical data potentially outdated.

## 3. How the Model Works

**3.1 Core Principle**

KNN regression predicts a target value (price) by identifying the `k` closest data points in the training set based on feature similarity and averaging their target values.

* Distance Metric: Euclidean distance was used , weighted by the inverse of distance, giving closer neighbors more influence.

**3.2 Cross-validation for Robustness**

K-fold cross-validation was implemented to assess the model's performance across multiple data splits, ensuring robust and reliable results.

**3.3 Model Evaluation Metrics**

* **Mean Absolute Error (MAE)**: MAE measures the average absolute difference between predicted and actual prices. It provides a straightforward interpretation of the model's error in the same units as the target (euro), making it practical for understanding the typical deviation in predictions.

* **Root Mean Squared Error (RMSE)**: RMSE penalizes larger errors more heavily due to squaring, offering a more sensitive measure of prediction accuracy. It is particularly useful for identifying significant outliers in property price predictions.

* **R-squared (R²)**: R² quantifies how well the model explains the variance in the target variable, with values closer to 1 indicating better fit. It helps assess the overall effectiveness of the model in capturing price trends.

* **Mean Absolute Percentage Error (MAPE)**: MAPE represents the average absolute error as a percentage of actual prices, allowing easy comparison of prediction accuracy across datasets with different price ranges. It is useful for understanding the relative error magnitude.

* **Symmetric MAPE (sMAPE)**: sMAPE accounts for symmetry in percentage errors, preventing overemphasis on either overestimation or underestimation. It’s particularly valuable for ensuring balanced performance in predicting property prices.

**3.4 Making the Model Interpretable**
To open the "black box" of the KNN model:

* **Feature Importance Analysis**: helps identify which features (e.g., living area, distance to cities) most influence predictions. Though KNN doesn't inherently provide feature importance, sensitivity analysis or visualization (e.g., partial dependence plots) can reveal their impact. This aids stakeholders in understanding what drives property prices.

* **Distance-based Analysis**: examines the average distance between test points and their nearest neighbors. This helps assess whether predictions are based on sufficiently local information or influenced by distant, less-relevant data, which is critical in ensuring accurate price estimates.

* **Plotting Predictions vs. True Values:** visualizes the alignment between predicted and actual prices. It highlights systematic errors, such as underprediction for high-priced properties, and helps refine the model by identifying regions or property types with high deviations.

## 4. Results and Insights

**4.1 Train Set Metrics**

* MAE (Mean Absolute Error: 4137.78): On average, the predicted property prices deviate by approximately €4,138 from the actual prices. This is a strong performance, indicating high accuracy on the training data.

* RMSE (Root Mean Squared Error: 14,208.59): While higher than the MAE due to the squared penalty on large errors, the RMSE still reflects a well-fitting model. The gap between MAE and RMSE is small, implying few extreme outliers in the training set.

* R² (Coefficient of Determination: 0.9904): The model explains 99.04% of the variance in property prices on the training set, indicating a nearly perfect fit.

* MAPE (Mean Absolute Percentage Error: 1.14%) and sMAPE (Symmetric MAPE: 1.12%): The low percentages indicate minimal error relative to the actual prices, further emphasizing the model's accuracy on the training data.

**4.2 Test Set Metrics**

* MAE (Mean Absolute Error: 51,453.95): On the test set, the average error is significantly higher. This large jump from the training set suggests potential overfitting or data mismatch between training and testing data.

* RMSE (Root Mean Squared Error: 79,155.82): The large RMSE reflects substantial deviations in some test predictions, likely caused by outliers or areas of the feature space that the model struggles with.

* R² (Coefficient of Determination: 0.7022): The model explains only 70.22% of the variance in test data. While this is a decent result, it highlights a performance drop compared to the training set.

* MAPE (Mean Absolute Percentage Error: 16.38%) and sMAPE (Symmetric MAPE: 14.79%): These values indicate a much larger relative error on the test set, emphasizing the need to address overfitting or test set inconsistencies.

**4.3 Insights**

* Overfitting: The high training performance versus the drop on the test set suggests that the model is potentially overfitting to the training data.

* Feature Space Coverage: The test set may include properties with characteristics not well-represented in the training set (e.g., rare property types or locations). This could cause poor generalization.

* Data Distribution: The test set might have a wider price range or include outliers, leading to higher prediction errors.

**4.4 Recommendations for Improvement**

* Regularization: Further feature selection or dimensionality reduction techniques (e.g., PCA) could be implemented to reduce noise in the dataset. Additional feature engineering could lead to improve the model's ability to generalize.

* Data Augmentation: A balancing of the dataset by adding more data from underrepresented property types or region could improve predictions as well as addressing further outliers through capping or transformation techniques.

## 5. Conclusion

The KNN regression model is a strong starting point for predicting real-estate prices due to its simplicity, transparency, and adaptability to localized trends. 

While the model has its limitations, careful data preprocessing and strategic optimizations can make it a valuable tool for this task. 

## Data Sources:

* Real-estate property data:
  * https://www.immoweb.be

* Statbel datasets for income and commune NIS-codes
  * https://statbel.fgov.be/en/themes/households/taxable-income#figures
  * https://statbel.fgov.be/nl/over-statbel/methodologie/classificaties/geografie