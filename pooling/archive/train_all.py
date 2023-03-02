import argparse
import os


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpus_avail', type = str, required = True)
    parser.add_argument('--gpu_file', type = str, required = True)
    args = parser.parse_args()

    countries = [i for i in os.listdir("../../imagery/") if ".ipynb" not in i]
    gpus_avail = [*args.gpus_avail]
    gpu_file = args.gpu_file

    for country in countries:
        
        with open(gpu_file, "r") as f:
            gpus_in_use = f.read().splitlines()
        print("gpus_in_use: ", gpus_in_use)
        
        if len(gpus_in_use) < len(gpus_avail):
            
            with open(gpu_file, "a") as f:
                f.append()
            
            
            # get training version for the current country
            cur_v = [int(i[-1]) for i in os.listdir("./") if country in i]
            if len(cur) != 0:
                v = max(cur) + 1
            else:
                v = 1

            print(cur, v)

            # make the subprocess command
            command = f"python3 train.py --folder_name {country}_v{v} --iso {country}"

            print(command)

        