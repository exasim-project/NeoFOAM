import argparse
import json
import subprocess
parser = argparse.ArgumentParser(description="combine profiling information into a file")
parser.add_argument("jsonFile", type=str, help="name of input json file")

args = parser.parse_args()
print(args)
with open(args.jsonFile, "r") as json_file:
    data = json.load(json_file)

for key,val in data.items():
    print("setting key: ",key, " to ", val)
    subprocess.run(["foamDictionary", "system/simulationParameters ", "-entry", f"{key}", "-set", f"{val}"])