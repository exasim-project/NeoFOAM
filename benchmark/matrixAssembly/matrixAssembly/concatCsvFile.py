import argparse
import pandas as pd
import os

parser = argparse.ArgumentParser(description="combine profiling information into a file")
parser.add_argument("nProcs", type=str, help="number of processors")
parser.add_argument("mesh", type=str, help="name of the mesh")

args = parser.parse_args()
csvs = []
for f in os.listdir():
    if "case_" in f and ".csv" in f:
        tmpdf = pd.read_csv(f)
        tmpdf["nProcs"] = args.nProcs
        tmpdf["mesh"] = args.mesh
        csvs.append(tmpdf)

df = pd.concat(csvs)

df.to_csv("results.csv",index=False)
