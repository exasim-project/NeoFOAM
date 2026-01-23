import foamadapter
import pytest

def test_greet():
    """Test that the C++ greet function is accessible from Python."""
    assert hasattr(foamadapter, "neofoam_bindings")
    assert foamadapter.neofoam_bindings.greet() == "Hello from NeoFOAM C++ bindings!"
