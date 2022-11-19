"""
Purpose: Test functions for ml module.
Author(s): Bobby Lumpkin
"""


import numpy as np
import pytest


from ml.model import (
    compute_model_metrics,
    inference,
    model,
    train_model
)


@pytest.mark.unittest
def test_train_model(
    categorical_features,
    label,
    data
):
    """
    Unittest for ml.train_model.
    """
    model_obj = train_model(
        train=data,
        categorical_features=categorical_features,
        label=label
    )
    assert isinstance(model_obj, model)


@pytest.mark.unittest
def test_compute_model_metrics(
    y,
    preds
):
    """
    Unittest for ml.compute_model_metrics.
    """
    metrics = compute_model_metrics(y, preds)
    assert len(metrics) == 3
    assert isinstance(metrics, tuple)


@pytest.mark.unittest
def test_inference(
    label,
    model,
    data
):
    """
    Unittest for ml.inference.
    """
    X = data.drop([label], axis=1)
    inference_results = inference(model=model, X=X)
    assert isinstance(inference_results, np.ndarray)
    assert len(inference_results) == len(data.index)


