# from shapely.geometry import Point
# import dask.dataframe as dd
# from shapely import wkt
# import geopandas as gpd
import pandas as pd
# import argparse
# import pygee
# import ee
import shutil
import os

if __name__ == "__main__":
    
    files = os.listdir("./data/clean/")
    files = [i for i in files if "test_" in i]
    print(files)

    dfs = []
    for i in files:
        dfs.append(pd.read_csv("./data/clean/" + i))
    
    df = pd.concat(dfs)
    
    print(df.shape)
    print(df)

    df.to_csv("./data/clean/lcs_unmapped.csv", index = False)
    
    for i in files:
        shutil.move("./data/clean/" + i, "./data/archive/" + i)