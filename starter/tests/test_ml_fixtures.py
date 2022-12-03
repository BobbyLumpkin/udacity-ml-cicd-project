"""
Purpose: Fixtures for test_ml.
Author(s): Bobby Lumpkin
"""


import joblib
import numpy as np
import pandas as pd
from pathlib import Path
import pytest


@pytest.fixture()
def categorical_features():
    return [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]


@pytest.fixture()
def label():
    return "salary"


@pytest.fixture()
def data():
    return pd.read_csv(
        Path(__file__).parents[1] / "data/census.csv"
    ).iloc[0:100]


@pytest.fixture()
def y():
    return np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])


@pytest.fixture()
def preds():
    return np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 0])


@pytest.fixture()
def model():
    model_path = Path(
        Path(__file__).parents[1] / "model/model_obj.pkl"
    )
    return joblib.load(model_path)

