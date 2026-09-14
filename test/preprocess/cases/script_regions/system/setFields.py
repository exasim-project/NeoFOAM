from neofoam.postprocess import Box, Sphere
from neofoam.preprocess import SetFields

setFields = SetFields(defaults={"alpha.water": 0.0})

setFields.assign(
    Box(min=(0, 0, -1), max=(0.1461, 0.292, 1)) | Sphere(center=(0, 0, 0), radius=0.25),
    {"alpha.water": 1.0},
)
