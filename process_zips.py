import geopandas as gpd
import pandas as pd
import zipfile
import os

def get_files(ext, direc):
    return [i for i in os.listdir(direc) if i.lower().endswith(ext)]
    
# Extract all of the zip files
for file in sorted(os.listdir("./data/zips/")):
    if ".ipynb" not in file:
        os.mkdir(f"./data/unzips/{file[:-4]}")
        with zipfile.ZipFile(f"./data/zips/{file}", 'r') as zip_ref:
            zip_ref.extractall(f"./data/unzips/{file[:-4]}")
            
            
wealth_dfs = []
shps = []
base_dir = "./data/unzips/"


for survey in os.listdir(base_dir):
    
    if ".ipynb" not in survey:

        try:
    
            f = os.path.join(base_dir, survey)

            gis = len(get_files(".shp", f))

            # Shapefiles
            if gis > 0:

                shp_path = get_files(".shp", f)[0]
                print(shp_path)
                gdf = gpd.read_file(os.path.join(base_dir, survey, shp_path))
                gdf["folder"] = survey
                shps.append(gdf)

            # Household recode files
            else:
                
                dta_path = get_files(".dta", f)[0]
                print(dta_path)
                dta = pd.read_stata(os.path.join(base_dir, survey, dta_path), convert_categoricals = False)
                dta.columns = [i.lower() for i in dta.columns]
                df_wealth = dta[["hv001", "hv007", "hv271"]]
                df_wealth["folder"] = survey
                wealth_dfs.append(df_wealth)
                
                        
        except Exception as e:
            
            with open("extract_errors_v4.txt", "a") as error_file:
                error_file.write(survey + ": " + str(e) + "\n")
                
                
# pd.concat(wealth_dfs).to_csv("./data/clean/wealth_data_v2.csv", index = False)                
gpd.pd.concat(shps).to_file("./data/clean/shps/point_data_v4.shp")
