import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ParameterGrid

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

from modules.modeling.metrics import ForecastMetrics


class ForecastPredictor:
    """Train, tune and generate forecasts for time series models."""

    def __init__(
        self,
        target_column="sales_quantity",
        random_state=42,
        n_jobs=-1
    ):
        self.target_column = target_column
        self.random_state = random_state
        self.n_jobs = n_jobs

    def split_time_series_data(
        self,
        data,
        validation_horizon=30,
        test_horizon=30
    ):
        train = data.iloc[:-(validation_horizon + test_horizon)].copy()
        validation = data.iloc[-(validation_horizon + test_horizon):-test_horizon].copy()
        test = data.iloc[-test_horizon:].copy()

        train_validation = pd.concat([train, validation], axis=0)

        return train, validation, test, train_validation

    def moving_average_forecast(self, history, target_index, window=7):
        forecast_value = history[self.target_column].iloc[-window:].mean()

        forecast = pd.Series(
            forecast_value,
            index=target_index,
            name=f"moving_average_{window}d_forecast"
        )

        return forecast

    def weighted_moving_average_forecast(self, history, target_index, window=7):
        recent_values = history[self.target_column].iloc[-window:]
        weights = np.arange(1, window + 1)

        forecast_value = np.average(recent_values, weights=weights)

        forecast = pd.Series(
            forecast_value,
            index=target_index,
            name=f"weighted_moving_average_{window}d_forecast"
        )

        return forecast

    def hybrid_baseline_forecast(
        self,
        history,
        target_index,
        seasonal_period=7,
        window=7,
        alpha=0.6
    ):
        forecast = pd.Series(index=target_index, dtype=float)
        history_series = history[self.target_column].copy()

        moving_average_value = history_series.iloc[-window:].mean()

        for date in target_index:
            lag_date = date - pd.Timedelta(days=seasonal_period)

            if lag_date in history_series.index:
                seasonal_value = history_series.loc[lag_date]
            else:
                seasonal_value = moving_average_value

            forecast.loc[date] = (
                alpha * seasonal_value +
                (1 - alpha) * moving_average_value
            )

        forecast.name = "hybrid_seasonal_moving_average_forecast"

        return forecast

    def tune_exponential_smoothing(self, y_train, y_validation):
        results = []

        trend_options = [None, "add"]
        seasonal_options = ["add"]
        seasonal_periods_options = [7, 14]

        for trend in trend_options:
            for seasonal in seasonal_options:
                for seasonal_periods in seasonal_periods_options:
                    try:
                        model = ExponentialSmoothing(
                            y_train,
                            trend=trend,
                            seasonal=seasonal,
                            seasonal_periods=seasonal_periods
                        ).fit(optimized=True)

                        forecast = model.forecast(len(y_validation))
                        forecast.index = y_validation.index

                        metrics = ForecastMetrics.calculate(
                            y_validation,
                            forecast
                        )

                        results.append({
                            "model": "Exponential Smoothing",
                            "trend": trend,
                            "seasonal": seasonal,
                            "seasonal_periods": seasonal_periods,
                            **metrics
                        })

                    except Exception as error:
                        print(
                            f"Error with trend={trend}, "
                            f"seasonal={seasonal}, "
                            f"seasonal_periods={seasonal_periods}: {error}"
                        )

        results_df = pd.DataFrame(results)
        best_params = results_df.sort_values("WAPE (%)").iloc[0]

        return best_params, results_df

    def fit_predict_exponential_smoothing(self, y_train, y_test, best_params):
        trend = best_params["trend"]

        if pd.isna(trend):
            trend = None

        seasonal = best_params["seasonal"]

        if pd.isna(seasonal):
            seasonal = None

        model = ExponentialSmoothing(
            y_train,
            trend=trend,
            seasonal=seasonal,
            seasonal_periods=int(best_params["seasonal_periods"])
        ).fit(optimized=True)

        forecast = model.forecast(len(y_test))
        forecast.index = y_test.index
        forecast.name = "exponential_smoothing_forecast"

        return model, forecast

    def tune_sarima(self, y_train, y_validation):
        results = []

        orders = [
            (1, 0, 1),
            (1, 1, 1),
            (2, 1, 1)
        ]

        seasonal_orders = [
            (1, 0, 1, 7),
            (1, 1, 1, 7)
        ]

        for order in orders:
            for seasonal_order in seasonal_orders:
                try:
                    model = SARIMAX(
                        y_train,
                        order=order,
                        seasonal_order=seasonal_order,
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(disp=False)

                    forecast = model.forecast(steps=len(y_validation))
                    forecast.index = y_validation.index

                    metrics = ForecastMetrics.calculate(
                        y_validation,
                        forecast
                    )

                    results.append({
                        "model": "SARIMA",
                        "order": order,
                        "seasonal_order": seasonal_order,
                        **metrics
                    })

                except Exception as error:
                    print(
                        f"Error with order={order}, "
                        f"seasonal_order={seasonal_order}: {error}"
                    )

        results_df = pd.DataFrame(results)
        best_params = results_df.sort_values("WAPE (%)").iloc[0]

        return best_params, results_df

    def fit_predict_sarima(self, y_train, y_test, best_params):
        model = SARIMAX(
            y_train,
            order=best_params["order"],
            seasonal_order=best_params["seasonal_order"],
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(disp=False)

        forecast = model.forecast(steps=len(y_test))
        forecast.index = y_test.index
        forecast.name = "sarima_forecast"

        return model, forecast

    def tune_random_forest(
        self,
        X_train,
        y_train,
        X_validation,
        y_validation,
        param_grid=None
    ):
        if param_grid is None:
            param_grid = {
                "n_estimators": [200, 300, 500],
                "max_depth": [3, 5, 8, None],
                "min_samples_leaf": [1, 3, 5],
                "max_features": ["sqrt", 0.7, 1.0]
            }

        results = []

        for params in ParameterGrid(param_grid):
            model = RandomForestRegressor(
                **params,
                random_state=self.random_state,
                n_jobs=self.n_jobs
            )

            model.fit(X_train, y_train)

            forecast = pd.Series(
                model.predict(X_validation),
                index=X_validation.index,
                name="random_forest_validation_forecast"
            )

            metrics = ForecastMetrics.calculate(
                y_validation,
                forecast
            )

            results.append({
                "model": "Random Forest",
                **params,
                **metrics
            })

        results_df = pd.DataFrame(results)
        best_params = results_df.sort_values("WAPE (%)").iloc[0]

        return best_params, results_df

    def fit_predict_random_forest(self, X_train, y_train, X_test, best_params):
        max_depth = best_params["max_depth"]

        if pd.isna(max_depth):
            max_depth = None
        else:
            max_depth = int(max_depth)

        model = RandomForestRegressor(
            n_estimators=int(best_params["n_estimators"]),
            max_depth=max_depth,
            min_samples_leaf=int(best_params["min_samples_leaf"]),
            max_features=best_params["max_features"],
            random_state=self.random_state,
            n_jobs=self.n_jobs
        )

        model.fit(X_train, y_train)

        forecast = pd.Series(
            model.predict(X_test),
            index=X_test.index,
            name="random_forest_forecast"
        )

        return model, forecast

    @staticmethod
    def get_feature_importance(model, feature_names):
        feature_importance = pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_
        })

        return feature_importance.sort_values(
            "importance",
            ascending=False
        )