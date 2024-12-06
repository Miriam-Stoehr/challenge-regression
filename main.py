from data_utils import DataUtils
from feature_utils import FeatureUtils
from model_utils import ModelUtils

def main():
    # Import data
    file_path = "./data/real_estate.csv"
    df = DataUtils.import_data(file_path)


    """Data Cleaning"""

    # Drop missing values in target column
    df = DataUtils.handle_missing_values(df, column='price', strategy='drop')

    # Fill missing values in 'facade_number' based on median per 'subtype_of_property'
    df = DataUtils.fill_missing_by_group(df, 'facade_number', group_column='subtype_of_property', agg_func='median')

    # Fill missing values in 'building_condition' with mode
    df = DataUtils.fill_missing_with_mode(df, 'building_condition', strategy='fill')

    # Drop outliers in price & garden (IQR), 'living_area (<=12), 'facade_number' (>=6), bedroom_nr (>=24) -> compared to other data points, don't seem realistic
    df = DataUtils.drop_outliers(df, 'price')
    df = DataUtils.drop_outliers(df, 'living_area', lower_bound=12)
    df = DataUtils.drop_outliers(df, 'facade_number', upper_bound=6)
    df = DataUtils.drop_outliers(df, 'bedroom_nr', upper_bound=24)

    # Remove substring 'unit' from 'subtype_of_property'
    df = DataUtils.remove_substring(df, 'subtype_of_property', ' unit')

    # Correct value for commune
    df = DataUtils.correct_value(df, column='commune', old_value='Petit-Rulx-lez-Nivelles', new_value='Petit-Rœulx-lez-Nivelles')


    """Data Encoding & Feature Engineering"""

    # Change 'terrace' to binary -> due to many instances when surface not given
    df = FeatureUtils.encode_binary(df, 'terrace', threshold=0)

    # Reduce categories for 'equipped_kitchen'
    df = FeatureUtils.map_manual(df, 'equipped_kitchen', {
        'installed': 'installed',
        'semi equipped': 'semi equipped',
        'hyper equipped': 'hyper equipped',
        'not installed': 'not installed',
        'usa installed': 'installed',
        'usa hyper equipped': 'hyper equipped',
        'usa semi equipped': 'semi equipped',
        'usa uninstalled': 'not installed',
        '0': 'not installed'
    })

    # Use Manual Ordinal Encoding on reduced categories of 'equipped_kitchen'
    df = FeatureUtils.encode_manual(df, 'equipped_kitchen', {
        'hyper equipped': 3,
        'installed': 2,
        'semi equipped': 1,
        'not installed': 0
    })

    # Use Manual Ordinal Encoding on 'building_condition'
    df = FeatureUtils.encode_manual(df, 'building_condition', {
        'as new': 5,
        'just renovated': 4,
        'good': 3,
        'to be done up': 2,
        'to renovate': 1,
        'to restore': 0
    })

    # Reduce categories for 'subtype_of property' by mapping
    df = FeatureUtils.map_manual(df, 'subtype_of_property', {
        'kot': 'apartment',
        'chalet': 'house',
        'flat studio': 'apartment',
        'service flat': 'apartment',
        'bungalow': 'house',
        'town house': 'house',
        'ground floor': 'apartment',
        'apartment': 'apartment',
        'house': 'house',
        'mixed use building': 'mixed use building',
        'triplex': 'house',
        'farmhouse': 'mixed use building',
        'loft': 'luxury',
        'duplex': 'house',
        'apartment block': 'other',
        'country cottage': 'house',
        'penthouse': 'luxury',
        'mansion': 'luxury',
        'other property': 'other',
        'villa': 'luxury',
        'exceptional property': 'luxury',
        'manor house': 'luxury',
        'castle': 'luxury'
    })

    # Use Manual Ordinal Encoding on 'subtype_of_property'
    df = FeatureUtils.encode_manual(df, 'subtype_of_property', {
        'luxury': 4,
        'other': 3,
        'house': 2,
        'mixed use building': 1,
        'apartment': 0
    })

    # Use Standard Ordinal Encoding on communes
    df = FeatureUtils.encode_ordinal(df, column='commune')


    """ Importing External Data"""

    # Import external datasets (refnis code and taxable income per commune)
    refnis_df = DataUtils.import_data("./data/postal_refnis_conv.xlsx")
    income_df = DataUtils.import_data("./data/taxable_income.xlsx")

    # Merge Refnis code and taxable income from external datasets in current df
    df = FeatureUtils.merge_data(df, refnis_df, import_col=['Postal code', 'Refnis code'], join_left='zip_code', join_right='Postal code')
    df = FeatureUtils.merge_data(df, income_df, import_col=['NIS code', 'Total net taxable income', 'Average values', 'Prosperity index'], join_left='Refnis code', join_right='NIS code')

    # Rename columns
    df = FeatureUtils.rename_columns(df, columns={'Refnis code': 'refnis_code', 'Total net taxable income': 'com_tot_income', 'Average values': 'com_avg_income', 'Prosperity index': 'com_prosp_index'})

    # Select features for ML model & split data
    target = 'price'
    features = ['commune_encoded', 'living_area', 'building_condition_encoded', 'terrace_encoded', 'equipped_kitchen_encoded', 'subtype_of_property_encoded', 'com_avg_income']


    """Assigning variables, splitting test and training set & standardizing features"""

    y = df[target].values  # Convert to NumPy array for compatibility with k-fold
    X = FeatureUtils.select_features(df, features=features).values

    # Perform k-fold cross-validation
    n_splits = 5
    n_neighbors = 5

    results = ModelUtils.cross_validate(X, y, n_splits=n_splits, n_neighbors=n_neighbors)
    metrics = results["metrics"]
    y_true = results["y_true"]
    y_pred = results["y_pred"]
    train_time = results["train_time"]
    inference_time = results["inference_time"]

    # Print metrics
    print("\nCross-Validation Metrics (Averaged):")
    for dataset, dataset_metrics in metrics.items():
        print(f"\n{dataset.capitalize()} Metrics:")
        for metric, value in dataset_metrics.items():
            print(f"{metric}: {value:.4f}")
    
    # Print training and inference times
    print(f"\nTraining Time: {train_time:.4f} seconds")
    print(f"Inference Time: {inference_time:.4f} seconds")

    # Plot predictions vs true values
    ModelUtils.plot_predictions(y_true, y_pred, title="Cross-Validation Predictions vs True Values")

    # Print a sample of predictions
    ModelUtils.print_sample_predictions(y_true, y_pred, num_samples=10)

if __name__ == "__main__":
    main()