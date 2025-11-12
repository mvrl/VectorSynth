import mercantile
import requests
import pandas as pd
import random
import os
from tqdm import tqdm

if __name__ == '__main__':
    df = pd.read_csv('./data/points/sydney_combined.csv')
    access_token = "<YOUR_MAPBOX_ACCESS_TOKEN>"

    zoom = 16

    for index, rows in tqdm(df.iterrows(), total=df.shape[0]):
        latitude = rows['latitude']
        longitude = rows['longitude']
        point_id = rows['point_id']

        # Get the tile coordinates for the specified latitude and longitude
        tile = mercantile.tile(longitude, latitude, zoom)  # Adjust the zoom level as needed

        #random_list = ['a', 'b', 'c']

        #ser = random.choice(random_list)

        # Satellite tile
        url = f"https://api.mapbox.com/v4/mapbox.satellite/{tile.z}/{tile.x}/{tile.y}@2x.jpg90?access_token={access_token}"
        output_filename = f"./data/sat_images/patch_{point_id}_{zoom}.jpeg"

        # OSM Tile
        # url = f"https://api.mapbox.com/styles/v1/cherd/cma5spbna00ab01s79ohd5ghb/tiles/512/{tile.z}/{tile.x}/{tile.y}?access_token={access_token}" 
        # output_filename = f"./data/osm_images/patch_{point_id}_{zoom}.jpeg"

        if os.path.isfile(output_filename):
            continue

        headers = {'Accept-Language': 'en', 'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:106.0) Gecko/20100101 Firefox/106.0'}

        try:
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            # make directory if not exists
            os.makedirs(os.path.dirname(output_filename), exist_ok=True)

            with open(output_filename, 'wb') as output_file:
                output_file.write(response.content)
            
        except requests.exceptions.RequestException as e:
            print(f'Error: {e}')