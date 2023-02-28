import pygee
import os

with open("./countries.txt", "r") as f:
    countries = f.read().splitlines()

base_dir = "shps"
    
for country in countries:
    if country not in os.listdir(base_dir):
        pygee.downloadGB(country, "0", base_dir)