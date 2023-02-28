from shapely.geometry import Polygon, Point
import geopandas as gpd
import pandas as pd
import shapely
import pyproj
import utm


if __name__ == "__main__":

    merged = gpd.read_file("./data/clean/shps/wealth_points.shp")
    merged["buffer_info"] = merged.apply(lambda x: buffer_point(x.geometry, x.DHSCC, x.buffer_siz), axis=1)
    merged[["box", 'utm_zone', "hemisphere"]] = pd.DataFrame(merged.buffer_info.tolist(), index= merged.index)
    merged = merged.drop(["buffer_info", "geometry"], axis = 1)
    merged = gpd.GeoDataFrame(merged, geometry = "box")
    merged.head()

    merged.to_file("./data/clean/shps/wealth_buffers.shp")

