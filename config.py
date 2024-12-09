# Defines variables for main.py

DATASET_FILE_PATH = "./data/real_estate_w_coordinates.csv"

JSON_FILE_PATH = "./data/city_coordinates.json"

REFNIS_FILE_PATH = "./data/postal_refnis_conv.xlsx"

INCOME_FILE_PATH = "./data/taxable_income.xlsx"

PREDICTIONS_FILE_PATH = "./graphs/predictions.png"

DISTANCES_FILE_PATH = "./graphs/avg_distances.png"

PERMUTATION_FILE_PATH = "./graphs/permutation_importance.png"

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

INCOME_IMP_COL = ["NIS code", "Average values"]

REFNIS_IMP_COL = ["Postal code", "Refnis code"]

EQUIPPED_KITCHEN_MAPPING = {
    "installed": "installed",
    "semi equipped": "semi equipped",
    "hyper equipped": "hyper equipped",
    "not installed": "not installed",
    "usa installed": "installed",
    "usa hyper equipped": "hyper equipped",
    "usa semi equipped": "semi equipped",
    "usa uninstalled": "not installed",
    "0": "not installed",
}

EQUIPPED_KITCHEN_ENCODING = {
    "hyper equipped": 3,
    "installed": 2,
    "semi equipped": 1,
    "not installed": 0,
}

BUILDING_CONDITION_ENCODING = {
    "as new": 5,
    "just renovated": 4,
    "good": 3,
    "to be done up": 2,
    "to renovate": 1,
    "to restore": 0,
}

SUBTYPE_MAPPING = {
    "kot": "apartment",
    "chalet": "house",
    "flat studio": "apartment",
    "service flat": "apartment",
    "bungalow": "house",
    "town house": "house",
    "ground floor": "apartment",
    "apartment": "apartment",
    "house": "house",
    "mixed use building": "mixed use building",
    "triplex": "house",
    "farmhouse": "mixed use building",
    "loft": "luxury",
    "duplex": "house",
    "apartment block": "other",
    "country cottage": "house",
    "penthouse": "luxury",
    "mansion": "luxury",
    "other property": "other",
    "villa": "luxury",
    "exceptional property": "luxury",
    "manor house": "luxury",
    "castle": "luxury",
}

SUBTYPE_ENCODING = {
    "luxury": 4,
    "other": 3,
    "house": 2,
    "mixed use building": 1,
    "apartment": 0,
}

RENAMED_COLUMNS = {"Refnis code": "refnis_code", "Average values": "com_avg_income"}
