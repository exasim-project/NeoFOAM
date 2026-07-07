# %%
# create 2DSquare and 3DCube benchmark cases

import sys
import os
import shutil
import subprocess
import pandas as pd
from pathlib import Path
from foamlib.preprocessing.parameter_study import record_generator
from foamlib.postprocessing.load_tables import datafile, load_tables
from foamlib.postprocessing.table_reader import read_catch2_benchmark


def build_records(name, resolution):
    return [
        {
            "case_name": f"{name}_N{str(r)}",
            "Res": r,
            "MeshType": name,
            "Resolution": f"N{r}",
        }
        for r in resolution
    ]


def create_cases(src, root, case, force=False, detailed=False):
    """creates benchmark cases"""
    case_path = root / case
    if case_path.exists() and not force:
        return
    if case_path.exists() and force:
        shutil.rmtree(case_path)
    case_path.mkdir(parents=True, exist_ok=True)

    cube_range = [8, 16, 32, 64, 128] if detailed else [16, 64]
    study_cube = record_generator(
        records=build_records("3DCube", cube_range),
        template_case=src / "templates/3DCube",
        output_folder=case_path / "Cases",
    )

    square_range = [8, 16, 32, 64, 128, 256, 512] if detailed else [64, 512]
    study_square = record_generator(
        records=build_records("2DSquare", square_range),
        template_case=src / "templates/2DSquare",
        output_folder=case_path / "Cases",
    )

    # Combine both studies
    final_study = study_cube + study_square
    final_study.create_study(study_base_folder=case_path)


def prepare_case(target, name):
    """After creating the benchmark folder make them executable"""
    r, dirs, fs = next(os.walk(target / name / "Cases"))
    root = Path(r)
    for study in dirs:
        dst = root / study / "0"
        src = root / study / "0.orig"
        if not dst.exists():
            shutil.copytree(src, dst)
        log = open(
            root / study / "blockMesh.log", "a"
        )  # so that data written to it will be appended
        proc = subprocess.Popen(
            ["blockMesh", ">", "blockMesh.log"],
            cwd=root / study,
            stdout=log,
            shell=True,
        )
        proc.wait()


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
        test_case_df = (
            test_case_df.groupby(group_keys)
            .apply(normalize_group, include_groups=False)
            .reset_index()
        )
        test_case_df.to_csv(results / f"{test_case}.csv", index=False)


def execute_case(executable, target, name):
    """Run the given executable over all target cases"""
    r, dirs, fs = next(os.walk(target / name / "Cases"))
    root = Path(r)
    for study in dirs:
        log = open(
            root / study / "execute.log", "a"
        )  # so that data written to it will be appended
        proc = subprocess.Popen(
            [executable, "--reporter", "xml", "-o", "stats.xml"],
            cwd=root / study,
            stdout=log,
            shell=False,
        )
        proc.wait()


def gather_results(target, name):
    results = target / name / "results"
    results.mkdir(exist_ok=True)

    cases = target / name / "Cases"
    if not cases.exists():
        print(f"could not find {cases}")

    file = datafile(file_name="stats.xml", folder=".")
    benchmark_results = load_tables(
        source=file, dir_name=cases, reader_fn=read_catch2_benchmark
    )
    if benchmark_results is not None:
        print(benchmark_results.columns)
        for test_case in benchmark_results["test_case"].unique():
            save_test_results(benchmark_results, test_case, results)
    else:
        print(f"failed {target}, {name}, {benchmark_results}")


def clean_case(target, name):
    """remove all logs"""
    r, dirs, fs = next(os.walk(target / name / "Cases"))


def display(target):
    cases = target
    r, dirs, fs = next(os.walk(cases))
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 1000)
    for f in fs:
        if not f.endswith("csv"):
            continue
        df = pd.read_csv(Path(r) / f)
        print(f"\n{f}")
        df["benchmark_name"] = df["benchmark_name"].apply(
            lambda x: x.replace("Executor", "")
        )
        df["Resolution"] = df["Resolution"].apply(lambda x: int(x[1:]))
        df["Cells"] = 0
        df.loc[df["MeshType"] == "2DSquare", "Cells"] = df["Resolution"] ** 2
        df.loc[df["MeshType"] == "3DCube", "Cells"] = df["Resolution"] ** 3
        df["Time/Cell"] = df["avg_runtime"] / df["Cells"]
        print(
            df.pivot(
                columns=["section1", "MeshType", "benchmark_name"],
                values="Time/Cell",
                index=["section2", "Resolution"],
            )
        )


def main():
    mode = sys.argv[1]
    modes = ["generate", "execute", "display", "clean"]
    if mode not in modes:
        print(f"{mode} not a valid modes, {modes}")
    if mode == "generate":
        src = sys.argv[2]
        target = sys.argv[3]
        case_name = sys.argv[4]
        detailed = sys.argv[5] == "fast"
        create_cases(Path(src), Path(target), case_name, True, detailed)
        prepare_case(Path(target), case_name)
    if mode == "execute":
        exe = sys.argv[2]
        target = sys.argv[3]
        case_name = sys.argv[4]
        execute_case(exe, Path(target), case_name)
        gather_results(Path(target), case_name)
    if mode == "display":
        target = sys.argv[2]
        display(Path(target))
    if mode == "clean":
        pass


if __name__ == "__main__":
    main()
