import pytest
from unittest.mock import MagicMock, patch

from vdaworld.core.api import WorldAPI

def test_api_dummy():
    # Test WorldAPI dummy initialization
    api = WorldAPI(output_dir="dummy_out")
    assert api.output_dir == "dummy_out"

