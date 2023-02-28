from shapely.geometry import Polygon, Point
import geopandas as gpd
import pandas as pd
import numpy as np
import shapely
import pyproj
import utm



def buffer_point(x1, x2, x3):
    
    # Get the point specific utm zone and projected coordinates
    x, y, zone, hem = utm.from_latlon(x1.y, x1.x)
    
    # Create the projection variables
    wgs84 = pyproj.CRS('EPSG:4326')
    utm_proj = pyproj.CRS.from_proj4(f"+proj=utm +datum=WGS84 +units=m +zone={str(zone)} +no_defs +ellps=WGS84 +towgs84=0,0,0")
    projection_back = pyproj.Transformer.from_crs(utm_proj, wgs84, always_xy=True).transform
    
    # Create the Point and buffer it
    p = gpd.GeoSeries([Point(x, y)]).buffer(x3 * 1000, cap_style = 3)

    # Project the polygon buffer back to 4326
    box = Polygon(shapely.ops.transform(projection_back, p[0]))
    
    return [box, zone, hem]


def gen_random_points(polygon, number):   
    minx, miny, maxx, maxy = polygon.bounds
    x = np.random.uniform( minx, maxx, number )
    y = np.random.uniform( miny, maxy, number )
    return list(zip(x, y))