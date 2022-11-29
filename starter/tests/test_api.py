"""
Purpose: Tests for API in main.py.
Author(s): Bobby Lumpkin
"""


from pathlib import Path
import pytest
import sys


sys.path.append(
    str(Path(__file__).parents[1])
)


@pytest.mark.unittest_api
def test_api_locally_get_root(client):
    r = client.get("/")
    assert r.status_code == 200