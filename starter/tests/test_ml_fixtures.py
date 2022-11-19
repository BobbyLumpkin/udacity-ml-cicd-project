"""
Purpose: Fixtures for test_ml.
Author(s): Bobby Lumpkin
"""


import joblib
import numpy as np
import pandas as pd
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
        "../data/census.csv"
    ).iloc[0:100]


@pytest.fixture()
def y():
    return np.array([1,1,1,1,1,1,1,1,1,1])


@pytest.fixture()
def preds():
    return np.array([1,1,1,1,1,1,1,1,1,0])


@pytest.fixture()
def model():
    model_path = (
        "/home/ubuntu/deploying-a-scalable-ml-pipeline-in-production"
        "/project/udacity-ml-cicd-project/starter/model/model_objs.pkl"
    )
    return joblib.load(model_path)

