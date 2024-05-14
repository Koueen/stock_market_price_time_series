import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def plot_heatmap(df):
    corr = df.corr().values
    fig = px.imshow(
        corr,
        x=df.columns,
        y=df.columns,
        color_continuous_scale='BuPu',
    )
    fig.update_xaxes(side='bottom')
    fig.update_traces(text=np.round(corr, 3), texttemplate='%{text}')
    fig.update_layout(
        title='Heatmap Correlation',
        title_x=0.5,
        title_y=0.93,
        width=1000,  # Set the width of the figure (in pixels)
        height=1000,  # Set the height of the figure (in pixels)
        autosize=False,  # Disable auto-sizing of the figure
    )
    fig.show()  # Tip: You can hover over all cells to see which pair of features denote that correlation cell.


def plot_stock_data(df):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.Close.index.values,
            y=df.Close.values,
            mode='lines',
            name='Close Stock Price',
            marker_color='rgb(56,41,131)',
        )
    )
    fig.add_trace(
        go.Scatter(  # Upper Bound
            name='High',
            x=df.High.index.values,
            y=df.High.values,
            mode='lines',
            marker=dict(color='#444'),
            line=dict(width=0),
            showlegend=False,
        )
    )
    fig.add_trace(
        go.Scatter(  # Lower Bound
            name='Low',
            x=df.Low.index.values,
            y=df.Low.values,
            marker=dict(color='#444'),
            line=dict(width=0),
            mode='lines',
            fillcolor='rgba(68, 68, 68, 0.3)',
            fill='tonexty',
            showlegend=False,
        )
    )
    fig.update_xaxes(title='days')
    fig.update_yaxes(title='Stock Price')
    fig.update_layout(
        hovermode='x',
        title='Maruti Suzuki India Stock Price',
        title_x=0.5,
        title_y=0.90,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1,
        ),
    )
    fig.show()


def plot_forecasting(
    original: pd.Series,
    fitted_data: pd.Series,
    predictions: pd.Series,
    title,
    zoom: bool = False,
    save: bool = False,
    n_days: int = 90,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=original.index.values,
            y=original.values,
            mode='lines',
            name='Original',
            marker_color='rgb(196,166,44)',
        )
    )
    fig.add_trace(
        go.Scatter(
            x=original.index.values,
            y=fitted_data,
            mode='lines',
            name='Fitted Train',
            marker_color='rgb(56,41,131)',
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predictions.index.values,
            y=predictions.values,
            mode='lines',
            name='Predictions',
            marker_color='rgb(180,230,122)',
        )
    )
    fig.update_xaxes(
        title='days',
        rangeslider_visible=True,
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="YTD", step="year", stepmode="todate"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
    )
    fig.update_yaxes(title='Stock Price')
    if zoom:
        new_ts = original.index[-1] + pd.Timedelta(days=n_days)
        xaxis_dict = dict(
            range=[datetime.datetime(2018, 1, 1), datetime.datetime(new_ts.year, new_ts.month, new_ts.day)]
        )
        yaxis = dict(range=[200, 600])
    else:
        xaxis_dict = dict()
        yaxis = dict()
    fig.update_layout(
        xaxis=xaxis_dict,
        yaxis=yaxis,
        title=title,
        title_x=0.5,
        title_y=0.90,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            x=1,
            xanchor='right',
        ),
    )
    if save:
        fig.write_image(f'images_result/forecast_{len(predictions)}_days.png')
    fig.show()


def get_metrics(predictions, test_set):
    # Test
    mse_e_test = mean_squared_error(test_set, predictions)
    mae_e_test = mean_absolute_error(test_set, predictions)

    print(pd.DataFrame({'Model': ['VAR'], 'Mode': ['Testing'], 'MSE': [mse_e_test], 'MAE': [mae_e_test]}))
