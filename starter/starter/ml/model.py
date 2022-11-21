from attrs import define, field
import joblib
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from typing import Union


from ml.data import process_data


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s:%(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


@define
class model_obj:
    encoder: OneHotEncoder
    lb: LabelBinarizer
    model: LGBMClassifier
    categorical_features: list = None
    label: str = None


@define
class model_metrics:
    model_obj: model_obj
    data: Union[pd.DataFrame, np.ndarray]
    slice_vars: Union[str, list] = None
    metrics_dict: dict = None
    
    def __attrs_post_init__(self):
        if not self.slice_vars:
            self.slice_vars = self.model_obj.categorical_features

    def compute_metrics(
        self,
        n_jobs: int = -1,
        verbose: int = 0,
        return_dict: bool = True
    ):
        """
        Compute model metrics and store as attributes.
        """
        # Process data and make predictions.
        _logger.info(
            "Processing data and generating predictions."
        )
        data = self.data
        model_obj = self.model_obj
        label = model_obj.label
        X, y, encoder, lb = process_data(
            data,
            categorical_features=model_obj.categorical_features,
            training=False,
            encoder=model_obj.encoder,
            label=label,
            lb=model_obj.lb
        )
        data["y"] = y
        data["preds"] = model_obj.model.predict(X)

        # Generate metrics for total population.
        _logger.info(
            "Generating metrics for total population."
        )
        metrics = compute_model_metrics(data["y"], data["preds"])
        metrics_dict = {
            "total_population" : {
                "precision" : metrics[0],
                "recall" : metrics[1],
                "fbeta" : metrics[2]
            }
        }

        def _compute_slice_var_metrics(data, slice_var, slice_val):
            """
            Compute metrics on a slice of a categorical variable.
            """
            data_slice = data[data[slice_var] == slice_val]
            y = data_slice.y
            preds = data_slice.preds
            metrics = compute_model_metrics(y, preds)
            return {
                "slice_val" : slice_val,
                "precision" : metrics[0],
                "recall" : metrics[1],
                "fbeta" : metrics[2]
            }
        
        for slice_var in self.slice_vars:
            _logger.info(
                "Generating metrics for {slice_var} "
                "slices.".format(slice_var=slice_var)
            )
            slice_vals = data[slice_var].unique().tolist()
            metrics_dict[slice_var] = Parallel(n_jobs=n_jobs, verbose=verbose)(
                delayed(_compute_slice_var_metrics)(
                    data=data,
                    slice_var=slice_var,
                    slice_val=slice_val
                )
                for slice_val in slice_vals
            )
        self.metrics_dict = metrics_dict
        if return_dict:
            return metrics_dict
        return


# Optional: implement hyperparameter tuning.
def train_model(
    train,
    categorical_features=[],
    save_path=None,
    label=None):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    save_path : str
        Path to save model to. (Optional)
    Returns
    -------
    model
        Trained machine learning model.
    """
    # Process training data.
    _logger.info("Processing training data.")
    X_train, y_train, encoder, lb = process_data(
        train,
        categorical_features=categorical_features,
        label=label,
        training=True
    )

    # Perform gridsearch over parameter space.
    n_estimators_list = list(range(100, 500, 100))
    num_leaves_list = list(range(2, 6))
    learning_rate_list = [0.1, 0.01, 0.001]
    parameters_lgbm = {
        'n_estimators': n_estimators_list,
        'num_leaves' : num_leaves_list,
        'learning_rate' : learning_rate_list
    }  
    lgbm_gs = GridSearchCV(
        LGBMClassifier(),
        parameters_lgbm,
        scoring="f1_macro",
        cv=5,
        verbose=1
    )
    _logger.info("Performing grid search over parameter space.")
    lgbm_gs.fit(X_train, y_train)
    _model = lgbm_gs.best_estimator_
    return_obj = model_obj(
        encoder=encoder,
        lb=lb,
        model=_model,
        categorical_features=categorical_features,
        label=label
    )
    _logger.info(
        "Final model is {_model}.".format(_model=_model)
    )
    
    # Save model, if appropriate.
    if save_path:
        _logger.info(
            "Saving model object to {save_path}.".format(save_path=save_path)
        )
        joblib.dump(return_obj, save_path)
    return return_obj


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model_obj, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model_obj : model_obj
        Saved model_objs, or path to.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Load model_objs if necessary.
    if isinstance(model_obj, str):
        _logger.info(
            "Loading model object from {model_obj}".format(model_obj=model_obj)
        )
        model_obj = joblib.load(model_obj)

    # Process data.
    _logger.info("Processing data for inference.")
    X_proc, y_proc, encoder, lb = process_data(
        X,
        categorical_features=model_obj.categorical_features,
        training=False,
        encoder=model_obj.encoder
    )
    _logger.info("Returning model scores.")
    return model_obj.model.predict(X_proc)

