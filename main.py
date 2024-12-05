from data_utils import DataUtils
from feature_utils import FeatureUtils
from model_utils import ModelUtils
from sklearn.model_selection import train_test_split

def main():
    # Import data
    file_path = "./data/real_estate.csv"
    df = DataUtils.import_csv(file_path)


    """Data Cleaning"""

    # Drop missing values in target column
    df = DataUtils.handle_missing_values(df, column='price', strategy='drop')

    # Fill missing values in 'facade_number' based on median per 'subtype_of_property'
    df = DataUtils.fill_missing_by_group(df, 'facade_number', group_column='subtype_of_property', agg_func='median')

    # Fill missing values in 'building_condition' with mode
    df = DataUtils.fill_missing_with_mode(df, 'building_condition', strategy='fill')

    # Drop outliers in price & garden (IQR), 'living_area (<=12), 'facade_number' (>=6), bedroom_nr (>=24) -> compared to other data points, don't seem realistic
    df = DataUtils.drop_outliers(df, 'price')
    df = DataUtils.drop_outliers(df, 'garden')
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

    # Select features for ML model & split data
    target = 'price'
    features = ['commune_encoded', 'living_area', 'building_condition_encoded', 'terrace_encoded', 'equipped_kitchen_encoded', 'subtype_of_property_encoded', 'garden']


    """Assigning variables, splitting test and training set & standardizing features"""

    y = df[target]
    X = FeatureUtils.select_features(df, features=features)

    # Convert data to float for standardization
    X = X.astype(float)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize Features
    X_train_scaled = ModelUtils.standardize_data(X_train, set_type='train')
    X_test_scaled = ModelUtils.standardize_data(X_test, set_type='test')


    """Training the Model, Making Predictions & Evaluating"""

    # Train the Model
    ModelUtils.train_model(X_train_scaled, y_train, n_neighbors=5)

    # Make Predictions
    y_train_pred = ModelUtils.predict(X_train_scaled)
    y_test_pred = ModelUtils.predict(X_test_scaled)

    # Evaluate on the training set
    train_metrics = ModelUtils.evaluate(y_train, y_train_pred)
    print("Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Evaluate on the test set
    test_metrics = ModelUtils.evaluate(y_test, y_test_pred)
    print("\nTest Metrics:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Plot predictions vs true values for the test set
    ModelUtils.plot_predictions(y_test, y_test_pred, title="Test Set Predictions vs True Values")

    # Print sample of predictions vs true values
    ModelUtils.print_sample_predictions(y_test, y_test_pred, num_samples=10)

if __name__ == "__main__":
    main()