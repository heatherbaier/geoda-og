import pandas as pd
import numpy as np
import jenkspy


if __name__ == "__main__":

    df = pd.read_csv("../data/clean/wealth_random_points.csv")

    dfs = []
    for iso in df["iso2"].unique():
        cur = df[df["iso2"] == iso]
        cur["mean_norm_hv271"] = (df["hv271"] - df["hv271"].mean()) / (df["hv271"].max() - df["hv271"].min())
        dfs.append(cur)
        print(iso, cur.shape)
        
    df = pd.concat(dfs)

    df.to_csv("../data/clean/wealth_random_points_mean_norm.csv", index = False)