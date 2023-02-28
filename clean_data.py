import geopandas as gpd
import pandas as pd
import numpy as np
import json


with open("./files_map.json", "r") as f:
    files_map = json.load(f)

# Clean wealth data
df = pd.read_csv("./data/clean/wealth_data.csv")
df["iso2"] = df["folder"].str[0:2]
wealth = pd.DataFrame(df.groupby(["folder", "hv001"])["hv271"].mean()).reset_index()
wealth["geo_folder"] = wealth["folder"].map(files_map)
wealth["hv001"] = wealth["hv001"].astype(str)
wealth.head()

# Clean spatial data
gdf = gpd.read_file("./data/clean/shps/point_data.shp")
gdf = gdf.rename(columns = {"folder": "geo_folder"})
gdf["iso2"] = gdf["DHSID"].str[0:2]
gdf["buffer_size"] = -99
gdf["buffer_size"] = np.where(gdf['URBAN_RURA'] == "U", 2, gdf["buffer_size"])
gdf["buffer_size"] = np.where(gdf['URBAN_RURA'] == "R", 5, gdf["buffer_size"])
gdf["hv007"] = gdf["DHSYEAR"].astype(int).astype(str)
gdf["hv001"] = gdf["DHSCLUST"].astype(int).astype(str)

# Merge wealth and spatial
merged = pd.merge(gdf, wealth, on = ["geo_folder", "hv001"])
merged["n_points"] = -99
merged["n_points"] = np.where(merged['URBAN_RURA'] == "U", 10, merged["n_points"])
merged["n_points"] = np.where(merged['URBAN_RURA'] == "R", 20, merged["n_points"])

merged.to_file("./data/clean/shps/wealth_points.shp")
