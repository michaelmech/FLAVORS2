"""Basic smoke tests for the flavors2 package.

These tests ensure that the package can be imported and that the
expected public API attributes exist. They are not comprehensive
unit tests but provide a quick check that the package structure is
correct and that the module is functional at a high level.
"""

def test_import():
    """Importing the top-level package should succeed."""
    import flavors2  # noqa: F401


def test_version_and_api():
    """The package should expose __version__ and FLAVORS2."""
    import flavors2

    assert hasattr(flavors2, "__version__")
    assert hasattr(flavors2, "FLAVORS2")