import pandas as pd
import numpy as np
import jenkspy


if __name__ == "__main__":

    df = pd.read_csv("../data/clean/wealth_random_points_jenks.csv")
    breaks = jenkspy.jenks_breaks(df['hv271'], n_classes = 4)
    df["wealth_class"] = -99
    df["wealth_class"] = np.where(df['hv271'] < breaks[1], 0, df["wealth_class"])
    df["wealth_class"] = np.where((df['hv271'] >= breaks[1]) & (df['hv271'] < breaks[2]), 1, df["wealth_class"])
    df["wealth_class"] = np.where((df['hv271'] >= breaks[2]) & (df['hv271'] < breaks[3]), 2, df["wealth_class"])
    df["wealth_class"] = np.where(df['hv271'] >= breaks[3], 3, df["wealth_class"])
    pd.concat(dfs).to_csv("../data/clean/wealth_data_global_jenks.csv", index = False)