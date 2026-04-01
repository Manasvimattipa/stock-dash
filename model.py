import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

def train_arima(close_prices, forecast_days):
    """
    close_prices: pandas Series with datetime index
    forecast_days: int
    """

    close_prices = close_prices.dropna()
    close_prices.index = pd.to_datetime(close_prices.index)

    model = ARIMA(close_prices, order=(5, 1, 0))
    model_fit = model.fit()

    forecast = model_fit.forecast(steps=forecast_days)

    last_date = close_prices.index[-1]
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=forecast_days,
        freq="B"
    )

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Forecast": forecast.values
    })

    return forecast_df
