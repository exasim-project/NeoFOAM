"""A script that forgets its TableSet — the loader cannot tell what to run."""

from neofoam.postprocess import VolIntegrate, field


def volume_p():
    return field("p") | VolIntegrate(name="volume_p")
