import argparse
import casefoam
import pandas as pd
import os

parser = argparse.ArgumentParser(description="combine profiling information into a file")
parser.add_argument("csvFile", type=str, help="name of output csv file")

args = parser.parse_args()
profData = []
for f in os.listdir("."):
    if "processor" in f:
        # get the profiling data of the correct processor
        tmpdf = casefoam.profiling(1,processorDir=f)
        tmpdf = tmpdf.rename(columns={'var_0': 'proc'})
        tmpdf["proc"] = f
        profData.append(tmpdf)

# combine alldata
data = pd.concat(profData)
data.to_csv(args.csvFile,index=False)