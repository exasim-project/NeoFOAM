# %%
# create 2DSquare and 3DCube benchmark cases

import sys
import os
import shutil
import subprocess
from pathlib import Path
from foamlib.preprocessing.parameter_study import record_generator
from foamlib.postprocessing.load_tables import datafile, load_tables
from foamlib.postprocessing.table_reader import read_catch2_benchmark

def build_records(name, resolution):
    return [ {"case_name": f"{name}_N{str(r)}", "Res":r, "MeshType": name, "Resolution": f"N{r}" } for r in resolution ]

def create_cases(src, root, case, force=False):
    """ creates benchmark cases"""
    case_path = root/case
    if case_path.exists() and not force:
        return

    case_path.mkdir(parents=True, exist_ok=True)
    study_cube = record_generator(
        records=build_records("3DCube", [8, 16, 32, 64, 128]),
        template_case= src / "templates/3DCube",
        output_folder=case_path / "Cases",
    )

    study_square = record_generator(
        records=build_records("2DSquare", [8, 16, 32, 64, 128, 256, 512]),
        template_case= src / "templates/2DSquare",
        output_folder=case_path / "Cases"
    )

    # Combine both studies
    final_study = study_cube + study_square
    final_study.create_study(study_base_folder=case_path)

def prepare_case(target, name):
    """ After creating the benchmark folder make them executable"""
    r, dirs, fs = next(os.walk(target/name/"Cases"))
    root = Path(r)
    for study in dirs:
        dst = root/study/"0"
        src = root/study/"0.orig"
        if not dst.exists():
            shutil.copytree(src, dst)
        log = open(root/study/"blockMesh.log",'a')  # so that data written to it will be appended
        subprocess.Popen(['blockMesh', '>', 'blockMesh.log'], cwd = root/study, stdout=log, shell=True)

def normalize_group(group):
    baseline = group.loc[group["benchmark_name"] == "OpenFOAM", "avg_runtime"]
    if not baseline.empty:
        group["normalized_speedup"] = baseline.values[0] / group["avg_runtime"]
    else:
        group["normalized_speedup"] = float("nan")  # No baseline found
    return group

# save per test case
def save_test_results(df, test_case: str, results):
    group_keys = ["MeshType", "Resolution"]
    test_case_df = df[df["test_case"] == test_case]
    if not test_case_df.empty:
        test_case_df = test_case_df.groupby(group_keys).apply(
            normalize_group, include_groups=False
        ).reset_index()
        test_case_df.to_csv(results / f"{test_case}.csv", index=False)

def execute_case(executable, target, name):
    """ Run the given executable over all target cases """
    r, dirs, fs = next(os.walk(target/name/"Cases"))
    root = Path(r)
    for study in dirs:
        log = open(root/study/"execute.log",'a')  # so that data written to it will be appended
        subprocess.Popen([executable, "--reporter", "xml", "-o", "stats.xml"], cwd = root/study, stdout=log, shell=False)

def gather_results(target, name):
    results = target / name / "results"
    results.mkdir(exist_ok=True)

    cases = target/ name / "Cases"
    if not cases.exists():
        print(f"could not find {cases}")

    file = datafile(file_name="stats.xml", folder=".")
    benchmark_results = load_tables(
        source=file, dir_name=cases, reader_fn=read_catch2_benchmark
    )
    try:
        for test_case in benchmark_results["test_case"].unique():
            save_test_results(benchmark_results, test_case, results)
    except Exception as e:
        print(f"Failed to postprocess {cases}, {e}")

def clean_case(target, name):
    """ remove all logs"""
    r, dirs, fs = next(os.walk(target/name/"Cases"))
    # for study in dirs:


def main():
    mode = sys.argv[1]
    if mode == "generate":
        src = sys.argv[2]
        target = sys.argv[3]
        case_name = sys.argv[4]
        create_cases(Path(src), Path(target), case_name)
        prepare_case(Path(target), case_name)
    if mode == "execute":
        exe = sys.argv[2]
        target = sys.argv[3]
        case_name = sys.argv[4]
        execute_case(exe, Path(target), case_name)
        gather_results(Path(target), case_name)
    if mode == "clean":
        pass

if __name__ == "__main__":
    main()
