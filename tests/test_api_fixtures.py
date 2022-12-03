"""
Purpose: Fixtures for test_api.py.
Author(s): Bobby Lumpkin
"""

from fastapi.testclient import TestClient
import pytest


@pytest.fixture()
def app():
    from main import app
    return app


@pytest.fixture()
def client(app):
    client = TestClient(app)
    return client


@pytest.fixture()
def post_scoring_path():
    return "/scoring/"