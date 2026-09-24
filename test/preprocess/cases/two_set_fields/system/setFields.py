from neofoam.preprocess import SetFields

setFields = SetFields(defaults={"alpha.water": 0.0})
alsoSetFields = SetFields(defaults={"alpha.air": 1.0})
