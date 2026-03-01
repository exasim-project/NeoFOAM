import neofoam


def test_import_neofoam_bindings() -> None:
    """Test that the C++ greet function is accessible from Python."""
    assert hasattr(neofoam, "neofoam_bindings")
