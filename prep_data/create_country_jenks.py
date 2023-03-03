import pandas as pd
import numpy as np
import jenkspy


if __name__ == "__main__":

    df = pd.read_csv("../data/clean/wealth_random_points_jenks.csv")
    
    dfs = []
    for iso in df["iso2"].unique():
        cur = df[df["iso2"] == iso]
        breaks = jenkspy.jenks_breaks(cur['hv271'], n_classes = 4)
        cur["wealth_class"] = -99
        cur["wealth_class"] = np.where(cur['hv271'] < breaks[1], 0, cur["wealth_class"])
        cur["wealth_class"] = np.where((cur['hv271'] >= breaks[1]) & (cur['hv271'] < breaks[2]), 1, cur["wealth_class"])
        cur["wealth_class"] = np.where((cur['hv271'] >= breaks[2]) & (cur['hv271'] < breaks[3]), 2, cur["wealth_class"])
        cur["wealth_class"] = np.where(cur['hv271'] >= breaks[3], 3, cur["wealth_class"])
        dfs.append(cur)
        print(iso, cur.shape, breaks)

    pd.concat(dfs).to_csv("../data/clean/wealth_data_country_jenks.csv", index = False)