# Script to train machine learning model.
import argparse
import logging
from matplotlib import pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

# Add the necessary imports for the starter code.
from ml.model import train_model, model_metrics


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

# Optional enhancement, use K-fold cross validation instead of a
# train-test split.
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
    "/project/udacity-ml-cicd-project/starter/model/model_obj.pkl"
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Model training and evaluation."
    )
    parser.add_argument(
        "--slice_plot_var", 
        type=str,
        help="Variable to generate slice plots for.",
        required=False,
        default="education"
    )
    args = parser.parse_args()

    # Process data, train and save model & encoder.
    _logger.info(
        "----------Training Model & Generating Model Obj----------"
    )
    train, test = train_test_split(data, test_size=0.20, random_state=123)
    model_obj = train_model(
        train=train,
        save_path=save_path,
        categorical_features=cat_features,
        label="salary"
    )
    
    # Generate model metrics and save slice metrics.
    _logger.info(
        "----------Generating Model Metrics----------"
    )
    metrics_obj = model_metrics(
        model_obj=model_obj,
        data=test
    )
    metrics_dict = metrics_obj.compute_metrics(return_dict=True)

    # Write metrics to 'slice_output.txt'.
    _logger.info(
        "----------Writing Metric Output to 'slice_output.txt'----------"
    )
    with open("../slice_output.txt", "w") as f:
        total_dict = metrics_dict["total_population"]
        total_pop_str = (
            "Total Population Metrics:\n"
            "    - precision = {precision}\n"
            "    - recall = {recall}\n"
            "    - fbeta = {fbeta}\n".format(
                precision=total_dict["precision"],
                recall=total_dict["recall"],
                fbeta=total_dict["fbeta"]
            )
        )
        f.write(total_pop_str)
        del metrics_dict["total_population"]
        for var, metrics in metrics_dict.items():
            var_str = "Slice metrics for {var}\n".format(var=var)
            f.write(var_str)
            for metric_dict in metrics:
                metrics_str = (
                    "    - slice = {slice_val}\n"
                    "      precision = {precision}\n"
                    "      recall = {recall}\n"
                    "      fbeta = {fbeta}\n".format(
                        slice_val=metric_dict["slice_val"],
                        precision=metric_dict["precision"],
                        recall=metric_dict["recall"],
                        fbeta=metric_dict["fbeta"]
                    )
                )
                f.write(metrics_str)
    
    # Generate & save slice plots
    _logger.info(
        "Generating and saving slice metric "
        "plots for {var}.".format(var=args.slice_plot_var)
    )
    var_slices = metrics_obj.metrics_dict[args.slice_plot_var]
    var_df = pd.DataFrame(var_slices)
    for metric in ["precision", "recall", "fbeta"]:
        f = plt.figure()
        f.set_figwidth(20)
        f.set_figheight(9)
        var_df.sort_values(
            by=metric,
            axis=0,
            ascending=False,
            inplace=True
        )
        plt.bar(
            x=var_df.slice_val,
            height=var_df[metric],
            width=0.8
        )
        plt.title(
            "{var} -- {metric}".format(
                var=args.slice_plot_var,
                metric=metric
            ),
            fontdict={"fontsize": 30}
        )
        plt.savefig(
            "../figures/{var}_slices_{metric}.png".format(
                var=args.slice_plot_var,
                metric=metric)
        )

