"""
Purpose: Tests for API in main.py.
Author(s): Bobby Lumpkin
"""


from fastapi.testclient import TestClient
import pytest


from main import app


# Instantiate the testing client with our app.
client = TestClient(app)

@pytest.mark.unittest_api
def test_api_locally_get_root():
    r = client.get("/")
    assert r.status_code == 200