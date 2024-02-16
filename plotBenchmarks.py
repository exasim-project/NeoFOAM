#%%
import xml.etree.ElementTree as ET
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Path to the XML file
xml_file = "blas.xml"

# Parse the XML file
tree = ET.parse(xml_file)

# Get the root element
root = tree.getroot()

#%%
def extract_benchmarks(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    testcase_name = ""
    section_names = []

    for testcase in root.iter('TestCase'):
        testcase_name = testcase.attrib['name']
        print(f'Test case: {testcase_name}')

        for section in testcase.iter('Section'):
            section_name = section.attrib['name']
            print(f'  Section: {section_name}')

            for benchmark in section.iter('BenchmarkResults'):
                for result in benchmark.iter('Result'):
                    name = result.attrib['name']
                    value = result.attrib['value']
                    print(f'    Benchmark {name}: {value}')

extract_benchmarks(xml_file)
#%%
def getBenchmarkData(element):
    for bRes in element:
        if bRes.tag == 'mean':
            meanValue = bRes.attrib['value']
            lowerBound = bRes.attrib['lowerBound']
            upperBound = bRes.attrib['upperBound']
        # if bRes.tag == 'mean':
        #     meanValue = bRes.attrib['value']
    return meanValue, lowerBound, upperBound

def process_element(f,element, testcase_name, section_names,depth=0):
    indent_str = ' ' * depth
    if element.tag == 'TestCase':
        testcase_name = element.attrib["name"]
    elif element.tag == 'Section':
        section_names[depth] = element.attrib["name"]
    elif element.tag == 'BenchmarkResults':
        bName = element.attrib['name']
        bValues = getBenchmarkData(element)
        f.write(f'{testcase_name},{",".join(section_names)},{bName},{",".join(bValues)}\n')

    for child in element:
        if element.tag == 'Section':
             process_element(f,child,testcase_name,section_names, depth + 1)
        else:
             process_element(f,child,testcase_name,section_names, depth + 0)

def nSections(element, depth=0):
    max_depth = max(depth, 0)
    for child in element:
        if element.tag == 'Section':
            max_depth = max(max_depth, nSections(child, depth + 1))
        else:
            max_depth = max(max_depth, nSections(child, depth + 0))

    return max_depth

def extract_benchmarks(xml_file, csv_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    nSec = nSections(root)
    section_names = [""]*nSec
    testcase_name = ""
    with open(csv_file, "w") as f:
        process_element(f,root,testcase_name,section_names)

extract_benchmarks(xml_file,"bench_blas.csv")

df  = pd.read_csv("bench_blas.csv", names=["Testcase","Section","Benchmark","mean","lowerBound","upperBound"])
print(df)
df["Section"] = pd.to_numeric(df["Section"], errors='coerce')

sns.scatterplot(data=df, x="Section", y="mean", hue="Benchmark")
plt.xscale("log")
plt.yscale("log")
plt.show()
# %%
