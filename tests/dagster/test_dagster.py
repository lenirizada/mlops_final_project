import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from dagster import build_op_context
from src.dagster.dags.main import iris_dataset, split_data, train_model, predict, log_to_mlflow


class TestIrisPipeline(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.mock_data = pd.DataFrame(
            np.random.rand(150, 4), columns=["feature1", "feature2", "feature3", "feature4"]
        )
        self.mock_data["target"] = np.random.randint(0, 3, 150)

    def test_iris_dataset(self):
        df = iris_dataset()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn("target", df.columns)

    def test_split_data(self):
        split = split_data(self.mock_data)
        self.assertIn("X_train", split)
        self.assertIn("X_test", split)
        self.assertIn("y_train", split)
        self.assertIn("y_test", split)
        self.assertEqual(len(split["X_train"]) + len(split["X_test"]), 150)

    def test_train_model(self):
        split = split_data(self.mock_data)
        model = train_model(split)
        self.assertIsInstance(model, RandomForestClassifier)

    def test_predict(self):
        split = split_data(self.mock_data)
        model = train_model(split)
        predictions = predict(model, split)
        self.assertIn("y_train_pred", predictions)
        self.assertEqual(len(predictions["y_train_pred"]), len(split["X_train"]))

    @patch("src.dagster.dags.iris_job.mlflow")
    @patch("src.dagster.dags.iris_job.plt.savefig")
    @patch("src.dagster.dags.iris_job.shap.TreeExplainer")
    def test_log_to_mlflow(self, mock_shap, mock_savefig, mock_mlflow):
        split = split_data(self.mock_data)
        model = train_model(split)
        predictions = predict(model, split)

        mock_shap.return_value.shap_values.return_value = np.random.rand(len(split["X_train"]), 4)
        mock_mlflow.start_run.return_value.__enter__.return_value.info.run_id = "test_run"
        mock_mlflow.register_model.return_value.name = "iris_classifier"
        mock_mlflow.register_model.return_value.version = 1

        context = build_op_context()
        log_to_mlflow(context, model, split, predictions)

        mock_mlflow.sklearn.log_model.assert_called()
        mock_mlflow.log_params.assert_called()
        mock_mlflow.log_metric.assert_called()
        mock_mlflow.log_artifact.assert_called()
        mock_mlflow.register_model.assert_called()


if __name__ == "__main__":
    unittest.main()