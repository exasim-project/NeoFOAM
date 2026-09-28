import neofoam


def test_import_neofoam_bindings() -> None:
    """Test that the compiled bindings module is importable from Python."""
    assert neofoam.neofoam_bindings is not None
