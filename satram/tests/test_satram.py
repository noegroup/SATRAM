"""
Unit and regression test for the satram package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import satram


def test_satram_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "satram" in sys.modules
