import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


class ForecastMetrics:
    """Calculate and compare forecasting metrics."""

    @staticmethod
    def calculate(y_true, y_pred):
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))

        denominator = np.sum(np.abs(y_true))

        if denominator == 0:
            wape = np.nan
        else:
            wape = (np.sum(np.abs(y_true - y_pred)) / denominator) * 100

        return {
            "MAE": round(mae, 2),
            "RMSE": round(rmse, 2),
            "WAPE (%)": round(wape, 2)
        }

    @staticmethod
    def evaluate_by_horizon(y_test, forecasts, horizons):
        results = []

        for horizon_name, horizon_steps in horizons.items():
            for model_name, forecast in forecasts.items():
                common_index = y_test.index.intersection(forecast.index)
                horizon_index = common_index[:horizon_steps]

                metrics = ForecastMetrics.calculate(
                    y_test.loc[horizon_index],
                    forecast.loc[horizon_index]
                )

                metrics["horizon"] = horizon_name
                metrics["n_days"] = len(horizon_index)
                metrics["model"] = model_name

                results.append(metrics)

        results_df = pd.DataFrame(results)

        return results_df[
            ["horizon", "n_days", "model", "MAE", "RMSE", "WAPE (%)"]
        ].sort_values(["horizon", "WAPE (%)"])