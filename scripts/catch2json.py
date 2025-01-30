# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2023-2025 NeoFOAM authors

import xmltodict
import json
import sys
import os


def parse_xml_dict(d):
    """takes the catch2 xml dict, performs clean-up and returns a
    list of records"""
    data = d["Catch2TestRun"]["TestCase"]
    if isinstance(data, dict):
        data = [data]
    records = []
    for cases in data:
        test_case = cases["@name"]
        for d in cases["Section"]:
            size = d["@name"]
            res = {}
            for k, v in d["BenchmarkResults"].items():
                # print(k, v)
                if k == "@name":
                    res["executor"] = v
                if k == "mean":
                    res["mean"] = v["@value"]
                    continue
                if k == "standardDeviation":
                    res["standardDeviation"] = v["@value"]
                    continue
                if k.startswith("@"):
                    continue
            res["size"] = size
            res["test_case"] = test_case
            records.append(res)
    return records


def main():
    _, _, files = next(os.walk("."))
    for xml_file in files:
        if not xml_file.endswith("xml"):
            continue
        try:
            with open(xml_file, "r") as fh:
                d = xmltodict.parse(fh.read())
                res = parse_xml_dict(d)
            with open(xml_file.replace("xml", "json"), "w") as outfile:
                json.dump(res, outfile)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
