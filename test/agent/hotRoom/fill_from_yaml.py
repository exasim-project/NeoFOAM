# %%
"""Fill the incompressibleFluid configs from ``configs/*.yaml`` and round-trip.

For every config class returned by ``configurations(incompressibleFluid)``:

1. **Validate** — load ``configs/<ClassName>.yaml`` and instantiate via
   ``cls(**data)``. Pydantic field validation and any ``model_validator``
   fire here.
2. **Write** — call ``.save(case_dir=OUT)`` so the registered IO strategy
   (OpenFOAM dict for ``constant/`` and ``system/`` files) materializes the
   instance on disk under this directory.
3. **Round-trip** — re-load with ``cls.load(case_dir=OUT)`` and compare
   ``model_dump()`` to the in-memory instance. This catches read/write
   asymmetries (the discriminated-union schemes in particular have custom
   serializers, so the round-trip is the real correctness check).

Run after editing any YAML in ``configs/`` to confirm it round-trips through
the OpenFOAM read/write path.
"""

import sys
from pathlib import Path
from pprint import pformat

import yaml

from neofoam.framework.solver.configurations import configurations
from neofoam.io import write_configs
from neofoam.solver.incompressibleFluid.incompressibleFluid import incompressibleFluid

OUT = Path(__file__).resolve().parent
CONFIGS = OUT / "configs"


def _trunc(obj: object, n: int = 240) -> str:
    s = pformat(obj, compact=True, width=120)
    return s if len(s) <= n else s[:n] + " …"


failed: list[str] = []
instances: dict[str, object] = {}  # name → validated instance, preserves order

# -----------------------------------------------------------------------------
# Phase 1: validate every YAML against its target class.
# -----------------------------------------------------------------------------
for cls in configurations(incompressibleFluid):
    name = cls.__name__
    yaml_path = CONFIGS / f"{name}.yaml"
    if not yaml_path.exists():
        print(f"SKIP  {name}: no {yaml_path.relative_to(OUT)}")
        continue

    data = yaml.safe_load(yaml_path.read_text()) or {}
    try:
        instances[name] = cls(**data)
    except Exception as e:  # noqa: BLE001
        failed.append(name)
        print(f"FAIL  {name}: validation: {type(e).__name__}: {e}")

# -----------------------------------------------------------------------------
# Phase 2: group by ``io_config.file`` and write each file once with all
# contributors merged. Without this, two classes targeting
# ``constant/transportProperties`` (e.g. TransportPropertiesConfig + Boussinesq)
# would clobber each other — the second save() does ``root_dict.clear()``.
# -----------------------------------------------------------------------------
with_io = [
    inst for inst in instances.values() if getattr(type(inst), "io_config", None)
]
try:
    report = write_configs(with_io, case_dir=OUT)
except Exception as e:  # noqa: BLE001
    print(f"FAIL  write_configs: {type(e).__name__}: {e}")
    failed.extend(type(c).__name__ for c in with_io)
    report = {}

for file, contribs in report.items():
    print(f"WRITE {file} ← {', '.join(contribs)}")

# -----------------------------------------------------------------------------
# Phase 3: round-trip each saved instance. After merge, the file holds keys
# from every contributor; ``cls.load`` only consumes its own declared fields,
# so the reloaded dump should match the original instance's dump.
# -----------------------------------------------------------------------------
for name, instance in instances.items():
    cls = type(instance)
    io = getattr(cls, "io_config", None)
    if io is None:
        print(f"OK    {name}: validated (no IO strategy → not written)")
        continue

    try:
        reloaded = cls.load(case_dir=OUT)
    except Exception as e:  # noqa: BLE001
        failed.append(name)
        print(f"FAIL  {name}: reload from {io.file}: {type(e).__name__}: {e}")
        continue

    before = instance.model_dump()
    after = reloaded.model_dump()
    if before == after:
        print(f"OK    {name}: round-trip clean")
    else:
        failed.append(name)
        print(f"FAIL  {name}: round-trip mismatch")
        print(f"  wrote:    {_trunc(before)}")
        print(f"  reloaded: {_trunc(after)}")

print(f"\n{len(failed)} failure(s)" + (f": {failed}" if failed else ""))

# -----------------------------------------------------------------------------
# Diagnostic: hard-set tutorial-shape values for Pimple_fvSchemes and write.
# Mirrors tutorials/hotRoom/system/fvSchemes — short OpenFOAM strings like
# "Gauss linear uncorrected" are unpacked into the discriminated-union scheme
# instances by the @BeforeValidator(_parse_*) parsers in neofoam.foam.schemes.
# -----------------------------------------------------------------------------
print("\n=== diagnostic: hard-set Pimple_fvSchemes → system/fvSchemes ===")

by_name = {cls.__name__: cls for cls in configurations(incompressibleFluid)}
Pimple_fvSchemes = by_name["Pimple_fvSchemes"]

HARDSET_SCHEMES: dict[str, dict[str, str]] = {
    "ddtSchemes": {"default": "Euler", "ddt(U)": "Euler"},
    "gradSchemes": {
        "default": "Gauss linear",
        "grad(U)": "Gauss linear",
        "grad(p)": "Gauss linear",
        "grad(p_rgh)": "Gauss linear",
        "grad(rhok)": "Gauss linear",
    },
    "divSchemes": {"default": "none", "div(phi,U)": "Gauss upwind"},
    "laplacianSchemes": {
        "default": "Gauss linear uncorrected",
        "laplacian(nuEff,U)": "Gauss linear uncorrected",
        "laplacian(rAU,p)": "Gauss linear uncorrected",
        "laplacian(rAUf,p_rgh)": "Gauss linear uncorrected",
    },
    "interpolationSchemes": {
        "default": "linear",
        "flux(HbyA)": "linear",
        "interpolate(rAU)": "linear",
        "dotInterpolate(S,U_0)": "linear",
        "flux(U)": "linear",
    },
    "snGradSchemes": {
        "default": "uncorrected",
        "snGrad(p)": "uncorrected",
        "snGrad(rhok)": "uncorrected",
        "snGrad(p_rgh)": "uncorrected",
    },
}

schemes = Pimple_fvSchemes.model_validate(HARDSET_SCHEMES)
schemes.save(case_dir=OUT)
fvSchemes_path = OUT / Pimple_fvSchemes.io_config.file
written = fvSchemes_path.read_text()
print(written)

missing = [
    f"{section}: {key}"
    for section, entries in HARDSET_SCHEMES.items()
    for key in entries
    if key not in written
]
if missing:
    failed.append("Pimple_fvSchemes (hard-set)")
    print(f"FAIL  hard-set Pimple_fvSchemes: missing on disk: {missing}")
else:
    print("OK    hard-set Pimple_fvSchemes: every declared entry present on disk")

print(f"\n{len(failed)} total failure(s)" + (f": {failed}" if failed else ""))
sys.exit(1 if failed else 0)
# %%
