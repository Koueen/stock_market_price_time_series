import json
import pickle
from datetime import datetime
from typing import List, Union

import numpy as np
import pandas as pd

from analysis.utils import plot_forecasting


class StockMarketPriceForecast:
    """Class used to forecast stock price"""

    def __init__(self, model_dir: str = 'model/var_model.pkl', data_dir: pd.DataFrame = 'data/ADANIPORTS.csv'):
        """Initialize StockMarketPriceForecast

        Args:
            model_dir (str): Path to model (VAR (Vector Autoregression)).
            data_dir (str): Path to data used for training.
        """
        self.models_dir = model_dir
        self.lag_order = 10
        self.data_dir = data_dir
        self.model = self.load_model()

    def __call__(self, n_days: int) -> pd.Series:
        """Predict the next n_days of close stock price

        Args:
            n_days (int): Number of days to forecast

        Returns:
            pd.Series: Index DateTime and stock price (float)
        """
        df = self.preprocess()  # This is used to plot to know from which point the forecasting is being predicted
        df_forecast = self.predict(n_days,df)
        plot_forecasting(df.Close,  self.model.fittedvalues.Close, df_forecast.Close, f'Forecasting next {n_days} days', True, True, n_days)
        return df_forecast.Close

    def preprocess(self) -> pd.DataFrame:
        """Preprocess raw csv into proper format

        Returns:
            pd.DataFrame: Composed by year | month | passengers. Their index is a datetime "year-month-day"
        """
        df = pd.read_csv(self.data_dir)
        df['Date'] = pd.to_datetime(df.Date, format='%Y-%m-%d')
        df.set_index('Date', inplace=True)
        df_roll = df[['Close','Turnover','Deliverable Volume']].rolling(7).mean().dropna()
        return df_roll

    def load_model(self):
        """Load model

        Returns:
            HoltWinters model
        """
        return pickle.load(open(self.models_dir, 'rb'))

    def predict(self, n_days: int, df: pd.DataFrame) -> pd.DataFrame:
        """Predict n_days

        Args:
            n_days: Number of days to predict in the future

        Returns:
            Returns:
            pd.Series: Index DateTime and stock market price (float)
        """
        date_range = pd.date_range(start='2020-01-1', periods=n_days, freq='D')
        forecast_input = df.values[-self.lag_order:]
        forecast = self.model.forecast(y=forecast_input,steps=n_days)
        df_forecast = pd.DataFrame(forecast, 
                           columns=df.columns, 
                           index=date_range)
        return df_forecast


if __name__ == '__main__':
    stockmarket_pr_obj = StockMarketPriceForecast()
    forecast = stockmarket_pr_obj(30)
    print(forecast)
