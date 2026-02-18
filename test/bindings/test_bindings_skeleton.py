import neofoam
import pytest

def test_greet():
    """Test that the C++ greet function is accessible from Python."""
    assert hasattr(neofoam, "neofoam_bindings")
    assert neofoam.neofoam_bindings.greet() == "Hello from NeoFOAM C++ bindings!"
