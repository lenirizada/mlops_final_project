import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import shap
from evidently.metric_preset import ClassificationPreset
from evidently.report import Report
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from dagster import Definitions, OpExecutionContext, asset, job

MLFLOW_TRACKING_URI = "http://mlflow:5000"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


@asset
def iris_dataset() -> pd.DataFrame:
    iris = load_iris(as_frame=True)
    df = iris.frame
    df["target"] = iris.target
    return df.head(20)


@asset
def split_data(iris_dataset: pd.DataFrame):
    X = iris_dataset.drop(columns=["target"])
    y = iris_dataset["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return {"X_train": X_train, "X_test": X_test, "y_train": y_train, "y_test": y_test}


@asset
def train_model(split_data):
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    mlflow.sklearn.log_model(model, "iris_rf_model")
    return model


@asset
def predict(train_model, split_data):
    X_train = split_data["X_train"]
    model = train_model

    y_train_pred = model.predict(X_train)
    return {"y_train_pred": y_train_pred}


@asset
def log_to_mlflow(context: OpExecutionContext, train_model, split_data, predict):
    X_train = split_data["X_train"]
    y_train = split_data["y_train"]
    model = train_model

    with mlflow.start_run():
        mlflow.sklearn.log_model(model, "iris_rf_model")
        mlflow.log_params({"n_estimators": 100, "random_state": 42})
        mlflow.log_metric("train_accuracy", model.score(X_train, y_train))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_train)
        shap.summary_plot(shap_values, X_train, show=False)
        plt.savefig("shap_summary.png")
        mlflow.log_artifact("shap_summary.png")

        report = Report(metrics=[ClassificationPreset()])
        report.run(
            reference_data=pd.concat([X_train, y_train], axis=1),
            current_data=pd.concat([X_train, y_train], axis=1),
        )
        report.save_html("evidently_report.html")
        mlflow.log_artifact("evidently_report.html")


@job
def iris_training_job():
    dataset = iris_dataset()
    split = split_data(dataset)
    model = train_model(split)
    predictions = predict(model, split)
    log_to_mlflow(model, split, predictions)


defs = Definitions(
    assets=[iris_dataset, split_data, train_model, predict],
    jobs=[iris_training_job],
)
