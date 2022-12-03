"""
Purpose: Tests for API in main.py.
Author(s): Bobby Lumpkin
"""


import json
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
    assert r.json() == {
        "greeting": (
            "Greetings!,\n\nWe hope you enjoy this API!"
            "\n\nBest,\nThe Creators"
        )
    }


@pytest.mark.unittest_api
@pytest.mark.parametrize("post_payload,expected", [
    ({
        'age': 31,
        'workclass': 'Private',
        'fnlgt': 45781,
        'education': 'Masters',
        'education_num': 14,
        'marital_status': 'Never-married',
        'occupation': 'Prof-specialty',
        'relationship': 'Not-in-family',
        'race': 'White',
        'sex': 'Female',
        'capital_gain': 14084,
        'capital_loss': 0,
        'hours_per_week': 50,
        'native_country': 'United-States'
    }, [1]),
    ({
        'age': 50,
        'workclass': 'Self-emp-not-inc',
        'fnlgt': 83311,
        'education': 'Bachelors',
        'education_num': 13,
        'marital_status': 'Married-civ-spouse',
        'occupation': 'Exec-managerial',
        'relationship': 'Husband',
        'race': 'White',
        'sex': 'Male',
        'capital_gain': 0,
        'capital_loss': 0,
        'hours_per_week': 13,
        'native_country': 'United-States'
    }, [0])
])
def test_api_locally_post_scoring(
    client,
    post_scoring_path,
    post_payload,
    expected
):
    r = client.post(post_scoring_path, data=json.dumps(post_payload))
    assert json.loads(r.content)["preds"] == expected