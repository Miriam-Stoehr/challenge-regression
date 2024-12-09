import json
from data_utils import DataUtils
from feature_utils import FeatureUtils
from model_utils import ModelUtils
from config import (
    DATASET_FILE_PATH,
    JSON_FILE_PATH,
    REFNIS_FILE_PATH,
    INCOME_FILE_PATH,
    PREDICTIONS_FILE_PATH,
    DISTANCES_FILE_PATH,
    PERMUTATION_FILE_PATH,
    CV_N_SPLITS,
    KNN_N_NEIGHBORS,
    TARGET,
    FEATURES,
    INCOME_IMP_COL,
    REFNIS_IMP_COL,
    EQUIPPED_KITCHEN_MAPPING,
    EQUIPPED_KITCHEN_ENCODING,
    BUILDING_CONDITION_ENCODING,
    SUBTYPE_MAPPING,
    SUBTYPE_ENCODING,
    RENAMED_COLUMNS,
)


def main():
    # Import dataset
    df = DataUtils.import_data(DATASET_FILE_PATH)

    # Load city coordinates from JSON file
    with open(JSON_FILE_PATH, "r") as json_file:
        city_coordinates_dict = json.load(json_file)

    """Data Cleaning"""

    # Drop missing values in target column
    df = DataUtils.handle_missing_values(df, column="price", strategy="drop")

    # Fill missing values in 'facade_number' based on median per 'subtype_of_property'
    df = DataUtils.fill_missing_by_group(
        df, "facade_number", group_column="subtype_of_property", agg_func="median"
    )

    # Fill missing values in 'building_condition' with mode
    df = DataUtils.fill_missing_with_mode(df, "building_condition", strategy="fill")

    # Drop outliers in price & garden (IQR), 'living_area (<=12), 'facade_number' (>=6), bedroom_nr (>=24) -> compared to other data points, don't seem realistic
    df = DataUtils.drop_outliers(df, "price")
    df = DataUtils.drop_outliers(df, "living_area", lower_bound=12)
    df = DataUtils.drop_outliers(df, "facade_number", upper_bound=6)
    df = DataUtils.drop_outliers(df, "bedroom_nr", upper_bound=24)

    # Remove substring 'unit' from 'subtype_of_property'
    df = DataUtils.remove_substring(df, "subtype_of_property", " unit")

    """Data Encoding & Feature Engineering"""

    # Change 'terrace' to binary -> due to many instances when surface not given
    df = FeatureUtils.encode_binary(df, "terrace", threshold=0)

    # Reduce categories for 'equipped_kitchen'
    df = FeatureUtils.map_manual(df, "equipped_kitchen", EQUIPPED_KITCHEN_MAPPING)

    # Use Manual Ordinal Encoding on reduced categories of 'equipped_kitchen'
    df = FeatureUtils.encode_manual(df, "equipped_kitchen", EQUIPPED_KITCHEN_ENCODING)

    # Use Manual Ordinal Encoding on 'building_condition'
    df = FeatureUtils.encode_manual(
        df, "building_condition", BUILDING_CONDITION_ENCODING
    )

    # Reduce categories for 'subtype_of property' by mapping
    df = FeatureUtils.map_manual(df, "subtype_of_property", SUBTYPE_MAPPING)

    # Use Manual Ordinal Encoding on 'subtype_of_property'
    df = FeatureUtils.encode_manual(df, "subtype_of_property", SUBTYPE_ENCODING)

    # Use Standard Ordinal Encoding on communes
    df = FeatureUtils.encode_ordinal(df, column="commune")

    """ Feature Engineering Based on External Data"""

    # Import external datasets (refnis code and taxable income per commune)
    refnis_df = DataUtils.import_data(REFNIS_FILE_PATH)
    income_df = DataUtils.import_data(INCOME_FILE_PATH)

    # Merge NIS code and taxable income from external datasets in current df
    df = FeatureUtils.merge_data(
        df,
        refnis_df,
        import_col=REFNIS_IMP_COL,
        join_left="zip_code",
        join_right=REFNIS_IMP_COL[0],
    )
    df = FeatureUtils.merge_data(
        df,
        income_df,
        import_col=INCOME_IMP_COL,
        join_left="Refnis code",
        join_right=INCOME_IMP_COL[0],
    )

    # Rename columns
    df = FeatureUtils.rename_columns(df, columns=RENAMED_COLUMNS)

    # Calculate distances to nearest of 10 largest cities of Belgium and add 'min_distance' column
    df = FeatureUtils.calculate_nearest_city_distance(df, city_coordinates_dict)

    """Split test and training set & standardize features"""
    # Assign target and feature variables (convert to NumPy array for compatibility with k-fold)
    y = df[TARGET].values  
    X = df[FEATURES].values

    # Train model & perform k-fold cross-validation
    results = ModelUtils.train_and_cross_validate(
        X, y, n_splits=CV_N_SPLITS, n_neighbors=KNN_N_NEIGHBORS
    )
    metrics = results["metrics"]
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    train_time = results["train_time"]
    inference_time = results["inference_time"]

    # Print metrics
    ModelUtils.print_metrics(metrics)

    # Print training and inference times
    print(f"\nTraining Time: {train_time:.4f} seconds")
    print(f"Inference Time: {inference_time:.4f} seconds")

    # Compute distance-based analysis
    X_scaled = ModelUtils._scaler.transform(X)
    avg_distances = ModelUtils.calculate_avg_neighbor_distances(X_scaled)

    # Compute permutation importance
    permutation_scores = ModelUtils.calc_permutation_importance(
        X_scaled, y, n_repeats=10
    )

    # Print a sample of predictions
    ModelUtils.print_sample_predictions(y_true, y_pred, num_samples=10)

    # Plot predictions vs true values
    ModelUtils.plot_predictions(y_true, y_pred, file_path=PREDICTIONS_FILE_PATH)

    # Plot distribution of average distances
    ModelUtils.plot_avg_neighbor_distances(avg_distances, file_path=DISTANCES_FILE_PATH)

    # Plot permutation importance
    ModelUtils.plot_permutation_importance(
        permutation_scores, FEATURES, file_path=PERMUTATION_FILE_PATH
    )

    # Print Notification
    print("Performance graphs were saved to '/graphs'.")


if __name__ == "__main__":
    main()
