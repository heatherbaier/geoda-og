import matplotlib.pyplot as plt
from urllib import request
import pandas as pd
import argparse
import os


API_KEY = ""


def GetGoogleStatic(lat, long, dhs_id, cumsum, iso):
    lat = str(lat)
    long = str(long)
    url = "https://maps.googleapis.com/maps/api/staticmap?center=" + str(lat) + "," + str(long) + "&zoom=16&size=400x400&maptype=satellite&key=" + API_KEY
    file = f"./imagery/{iso}/{dhs_id}_{cumsum}.png"
    request.urlretrieve(url, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--iso', type = str, required = True)
    args = parser.parse_args()    

    coords = pd.read_csv("./wealth_random_points.csv")
    coords["longitude"] = coords["random_points"].str.split(",").str[0].str.replace("(", "").astype(float)
    coords["latitude"] = coords["random_points"].str.split(",").str[1].str.replace(")", "").astype(float)
    coords["iso2"] = coords["DHSID"].str[0:2]
    coords = coords[coords["iso2"] == args.iso]
    print(coords.head())

    count = 0

    for index, row in coords.iterrows():

        try:
        
            file = f"./imagery/{args.iso}/{row['DHSID']}_{row['cumsum']}.png"
            
            if not os.path.exists(file):
            
                msg = "File #" + str(count)
                GetGoogleStatic(row['latitude'], row['longitude'], row['DHSID'], row['cumsum'], args.iso)
                count += 1

                with open(f"./{args.iso}_counter.txt", "w") as f:
                    f.write(str(count) + " out of " + str(len(coords)))

        except:

            pass
        
    