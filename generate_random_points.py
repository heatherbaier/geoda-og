import geopandas as gpd
import numpy as np

from utils import *


if __name__ == "__main__":
    
    gdf = gpd.read_file("./buffer_test.shp")
    gdf['random_points'] = gdf.apply(lambda x: gen_random_points(x['geometry'], x['n_points']), axis = 1)
    gdf = gdf[["DHSID", "DHSYEAR", "DHSCLUST", "folder", "buffer_siz", "hv001", "hv271", "geometry", "n_points", "random_points"]]
    gdf = gdf.explode('random_points')
    gdf["val"] = 1
    gdf['cumsum'] = gdf[["DHSID", 'val']].groupby('DHSID').cumsum()

    gdf.to_csv("./data/clean/wealth_random_points.csv", index = False)