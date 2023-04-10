from shapely.geometry import Point
import dask.dataframe as dd
from shapely import wkt
import geopandas as gpd
import pandas as pd
import argparse
import pygee
import ee



def get_lc_type(x):
    
    x = x["geom"]
        
    p = Point(x[0], x[1]).buffer(1)
    
    t = pygee.convert_to_ee_feature(p)
    
    temp = imagery.filterBounds(t)
    
    toRet = temp.first().reduceRegion(
        reducer = ee.Reducer.mode(),
        geometry = t
    ).get('LC_Type1').getInfo()
        
    return toRet


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
#     parser.add_argument('--folder_name', type = str, required = True)
    parser.add_argument('--iso', type = str, required = True)
#     parser.add_argument('--gpu', type = str, required = True)
    args = parser.parse_args()
    

    ee.Initialize()

    imagery = ee.ImageCollection('MODIS/006/MCD12Q1').filterDate('2020-01-01', '2020-01-02')#.filterBounds(cur_shp)

    gdf = pd.read_csv("./data/clean/wealth_data_country_jenks.csv")
    gdf = gdf[gdf["iso2"] == args.iso]#.sample(10)
    gdf["lng"] = gdf["random_points"].str.split(", ").str[0].str.replace("(", "").astype(float)
    gdf["lat"] = gdf["random_points"].str.split(", ").str[1].str.replace(")", "").astype(float)
    gdf["geom"] = list(zip(gdf["lng"], gdf["lat"]))
    gdf["geometry"] = gpd.GeoSeries.from_wkt(gdf["geometry"])
    gdf = gpd.GeoDataFrame(gdf, geometry = "geometry")
    ddf = dd.from_pandas(gdf, npartitions = 64)
    q = ddf.apply(get_lc_type, axis=1, meta=(None, 'int64'))
    lc = q.compute()
    gdf["lc_type1"] = lc

    gdf.to_csv(f"./data/clean/test_{args.iso}.csv", index = False)

    print(gdf.head())

    print(gdf["lc_type1"])