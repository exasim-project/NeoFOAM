from neofoam.postprocess import Box
from neofoam.preprocess import SetFields

setFields = SetFields(defaults={"alpha.water": 0.0})

# The right-hand half of the unit box; the case's setFields.yaml region (the
# left-hand half) is appended after this one.
setFields.assign(Box(min=(0.5, 0, -1), max=(1, 1, 1)), {"alpha.water": 0.25})
