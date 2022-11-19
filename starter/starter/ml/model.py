from attrs import define, field
import joblib
from lightgbm import LGBMClassifier
import logging
from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import OneHotEncoder


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
class model:
    encoder: OneHotEncoder
    model: LGBMClassifier
    categorical_features: list = None
    label: str = None


# Optional: implement hyperparameter tuning.
def train_model(
    train,
    save_path,
    categorical_features=[],
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
    return_obj = model(
        encoder=encoder,
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


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model : model
        Saved model_objs, or path to.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    # Load model_objs if necessary.
    if isinstance(model, str):
        _logger.info(
            "Loading model object from {model}".format(model=model)
        )
        model = joblib.load(model)

    # Process data.
    _logger.info("Processing data for inference.")
    X_proc, y_proc, encoder, lb = process_data(
        X,
        categorical_features=model.categorical_features,
        training=False,
        encoder=model.encoder
    )
    _logger.info("Returning model scores.")
    return model.model.predict(X_proc)

