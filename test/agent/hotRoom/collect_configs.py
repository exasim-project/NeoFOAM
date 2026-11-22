# %%
import json
from pathlib import Path

from neofoam.framework.solver.configurations import configurations
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid

OUT = Path(__file__).resolve().parent
SCHEMAS = OUT / "schemas"
SCHEMAS.mkdir(exist_ok=True)

for cls in configurations(incompressibleFluid):
    schema_path = SCHEMAS / f"{cls.__name__}.schema.json"
    schema_path.write_text(json.dumps(cls.model_json_schema(), indent=2) + "\n")
    print(f"wrote {schema_path.relative_to(OUT)}")
    if getattr(cls, "io_config", None) is None:
        continue
    cls.model_construct().save(case_dir=OUT)
    print(f"wrote {cls.io_config.file}")

# %%
