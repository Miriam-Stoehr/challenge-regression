from typing import Tuple, Optional, Dict, List
from geopy.geocoders import Nominatim
import json
import pandas as pd
import time
import os

def main() -> None:
    """
    Main function to get coordinates for locations in a dataframe and of cities provided in a list and save results to CSV.
    """
    # Ensure the output directory exists
    os.makedirs("./data", exist_ok=True)

    # Initialize raw data
    raw_data_path = "./data/real_estate.csv"
    output_path = "./data/real_estate_w_coordinates.csv"
    df = pd.read_csv(raw_data_path)
    column= 'commune'
    cities = [
    'Brussels', 'Antwerp', 'Ghent', 'Charleroi', 'Liège',
    'Anderlecht', 'Schaarbeek', 'Bruges', 'Namur', 'Leuven', 
    'Molenbeek-Saint-Jean', 'Mons'
    ]

    # Check column
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in the dataset.")

    # Initialize GeoCoder
    geo_coder = GeoCoder(user_agent="geoapi")

    # Get location coordinates
    print("Starting to fetch coordinates for locations.")
    coordinates_manager = CoordinatesManager(dataframe=df, column=column, geocoder=geo_coder)
    updated_df = coordinates_manager.add_coordinates_to_dataframe()

    # Save updated DataFrame to CSV
    updated_df.to_csv(output_path, index=False)

    # Fetch city coordinates
    print("\nStarting to fetch coordinates for cities.")
    city_fetcher = CityCoordinatesFetcher(geocoder=geo_coder)
    city_coordinates_dict = city_fetcher.get_coordinates(city_list=cities)

    # Save city_coordinates_dict to a JSON file
    with open("./data/city_coordinates.json", "w") as json_file:
        json.dump(city_coordinates_dict, json_file, indent=4)

    print("City coordinates have been saved to ./data/city_coordinates.json")

class GeoCoder:
    """
    A class to handle geocoding operations using the Geopy library.
    """

    def __init__(self, user_agent: str) -> None:
        """
        Initializes the GeoCoder with a user agent.
        
        :param user_agent: A unique identifier for the geocoder instance.
        """
        self.geolocator = Nominatim(user_agent=user_agent)

    def get_lat_lon(self, location_name: str) -> Tuple[Optional[float], Optional[float]]:
        """
        Retrieves the latitude and longitude for a given location name.
        
        :param location_name: The name of the location to geocode.
        :return: A tuple containing latitude and longitude, or (None, None) if not found.
        """
        try:
            location = self.geolocator.geocode(f"{location_name}, Belgium")
            if location:
                print(f"Fetching coordinates for {location_name}.")
                return location.latitude, location.longitude
        except Exception as e:
            print(f"Error geocoding {location_name}: {e}")
        return None, None


class CoordinatesManager:
    """
    A class to manage latitude and longitude extraction for locations.
    """

    def __init__(self, dataframe: pd.DataFrame, column: str, geocoder: GeoCoder) -> None:
        """
        Initializes the CoordinatesManager.
        
        :param dataframe: The DataFrame containing location data.
        :param column: The column with locations to get coordinates for.
        :param geocoder: An instance of GeoCoder for geocoding locations.
        """
        self.dataframe = dataframe
        self.column = column
        self.geocoder = geocoder

    def add_coordinates_to_dataframe(self) -> pd.DataFrame:
        """
        Adds latitude and longitude columns to the DataFrame.
        
        :return: The updated DataFrame with latitude and longitude.
        """
        locations = sorted(self.dataframe[self.column].unique())  # Sort locations in alphabetical order for user-friendly notification
        coordinates = [
            self.geocoder.get_lat_lon(location) for location in locations
        ]
        location_lat_lon_df = pd.DataFrame(
            {self.column: locations, 'latitude': [c[0] for c in coordinates], 'longitude': [c[1] for c in coordinates]}
        )
        return self.dataframe.merge(location_lat_lon_df, on=self.column, how='left')


class CityCoordinatesFetcher:
    """
    A class to fetch and store coordinates of cities.
    """

    def __init__(self, geocoder: GeoCoder, pause_duration: int = 1) -> None:
        """
        Initializes the CityCoordinatesFetcher.
        
        :param geocoder: An instance of GeoCoder for geocoding cities.
        :param pause_duration: Time to pause between geocoding requests (in seconds).
        """
        self.geocoder = geocoder
        self.pause_duration = pause_duration

    def get_coordinates(self, city_list: List[str]) -> Dict[str, Optional[Tuple[float, float]]]:
        """
        Retrieves the coordinates of a list of cities.
        
        :param city_list: A list of city names.
        :return: A dictionary mapping city names to their coordinates.
        """
        city_list = sorted(city_list) # Sort cities in alphabetical order for user-friendly notification
        city_coordinates = {}
        for city in city_list:
            lat_lon = self.geocoder.get_lat_lon(city)
            city_coordinates[city] = lat_lon
            time.sleep(self.pause_duration)
        return city_coordinates

if __name__ == "__main__":
    main()
