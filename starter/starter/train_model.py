# Script to train machine learning model.
import logging
import pandas as pd

# Add the necessary imports for the starter code.
from ml.model import train_model, inference, compute_model_metrics


_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)
_formatter = logging.Formatter(
    "%(asctime)s:%(levelname)s:%(module)s:%(message)s"
)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(_formatter)
_logger.addHandler(_console_handler)


# Add code to load in the data.
data = pd.read_csv(
    "../data/census.csv"
)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

save_path = (
    "/home/ubuntu/deploying-a-scalable-ml-pipeline-in-production"
    "/project/udacity-ml-cicd-project/starter/model/model_objs.pkl"
)


if __name__ == "__main__":
    # Process data, train and save model & encoder.
    model = train_model(
        train=data,
        save_path=save_path,
        categorical_features=cat_features,
        label="salary"
    )

