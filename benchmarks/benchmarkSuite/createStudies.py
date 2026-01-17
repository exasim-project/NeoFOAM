# %%
# create 2DSquare and 3DCube benchmark cases

from pathlib import Path
from foamlib.preprocessing.parameter_study import record_generator

def build_records(name, resolution):
    return [ {"case_name": f"{name}_N{str(r)}", "Res":r, "MeshType": name, "Resolution": f"N{r}" } for r in resolution ]

def create_cases(root, case):
    case_path = root/case
    case_path.mkdir(parents=True, exist_ok=True)
    study_cube = record_generator(
        records=build_records("3DCube", [8, 16, 32, 64, 128, 200]),
        template_case=root / "templates/3DCube",
        output_folder=case_path / "Cases",
    )

    study_square = record_generator(
        records=build_records("2DSquare", [8, 16, 32, 64, 128, 256, 512, 1024, 2048]),
        template_case=root / "templates/2DSquare",
        output_folder=case_path / "Cases"
    )

    # Combine both studies
    final_study = study_cube + study_square
    final_study.create_study(study_base_folder=case_path)

root = Path(__file__).parent
# for c in ["explicitOperators",  "implicitOperators", "dsl"]:
#     create_cases(root, c)
create_cases(root, "dsl")