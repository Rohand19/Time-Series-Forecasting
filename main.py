import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from nixtla import NixtlaClient
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.graph_objects as go
import asyncio
from dotenv import load_dotenv
import os
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import warnings

warnings.filterwarnings('ignore')
load_dotenv()

# Function to calculate MAPE
def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    non_zero = (y_true != 0)
    return np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100

# Function to calculate SMAPE
def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    numerator = np.abs(y_pred - y_true)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2

    # Handle division by zero
    mask = denominator != 0
    valid_entries = numerator[mask] / denominator[mask]

    return np.mean(valid_entries) * 100

# Function to calculate MASE
def mean_absolute_scaled_error(y_true, y_pred, y_train):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    n = len(y_train)
    d = np.abs(np.diff(y_train)).sum() / (n - 1)
    if d == 0:
        d = 1e-6

    errors = np.abs(y_true - y_pred)
    return errors.mean() / d

# Function to run Prophet model
def run_prophet(df, forecast_horizon):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
    model.fit(df)
    future = model.make_future_dataframe(periods=forecast_horizon)
    forecast = model.predict(future)
    return forecast

async def run_timegpt(df, **kwargs):
    forecast = st.session_state.timegpt.forecast(df, **kwargs)
    return forecast

# Function to run ARIMA model
def run_arima(df, forecast_horizon):
    model = ARIMA(df['y'], order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=forecast_horizon)
    return forecast

# Function to run SARIMA model
def run_sarima(df, forecast_horizon):
    try:
        model = SARIMAX(df['y'], order=(1,1,1), seasonal_order=(1,1,1,12))
        results = model.fit()
        forecast = results.forecast(steps=forecast_horizon)
    except:
        st.warning("SARIMA model failed to converge. Falling back to ARIMA model.")
        forecast = run_arima(df, forecast_horizon)
    return forecast

# Function to run Exponential Smoothing (Holt-Winters) model
def run_exponential_smoothing(df, forecast_horizon):
    model = ExponentialSmoothing(df['y'], seasonal='add', seasonal_periods=12)
    results = model.fit()
    forecast = results.forecast(steps=forecast_horizon)
    return forecast

# Function to evaluate models
def evaluate_model(actual, predicted):
    # Ensure actual and predicted have the same length
    min_len = min(len(actual), len(predicted))
    actual = actual[:min_len]
    predicted = predicted[:min_len]

    mae = mean_absolute_error(actual, predicted)
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    rmae = np.sqrt(mae)
    smape = symmetric_mean_absolute_percentage_error(actual, predicted)
    mape = mean_absolute_percentage_error(actual, predicted)
    mase = mean_absolute_scaled_error(actual, predicted, actual)
    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'MAPE': mape,
        'MASE': mase,
        "RMAE": rmae,
        'SMAPE': smape
    }

# Streamlit app
def main():
    st.set_page_config(
        page_title='Time Series Forecasting App',
        page_icon='📈',
        layout='wide',
        initial_sidebar_state='collapsed'
    )
    st.title('Time Series Forecasting App')

    if "timegpt" not in st.session_state:
        st.session_state.timegpt = NixtlaClient(os.getenv('NIXTLA_API_KEY'))
        st.session_state.timegpt.validate_api_key()

    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.dataframe(df.head(), use_container_width=True)

        # Select date and value columns
        date_col = st.selectbox('Select Date Column', df.columns)
        value_col = st.selectbox('Select Value Column', df.columns)

        # Prepare data for models
        df['ds'] = pd.to_datetime(df[date_col])
        df['y'] = df[value_col]
        df = df[['ds', 'y']].sort_values('ds')

        # Plot original time series
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Original Data'))
        fig.update_layout(title='Time Series Plot', xaxis_title='Date', yaxis_title='Value')
        st.plotly_chart(fig)

        # Forecast horizon
        forecast_horizon = st.slider('Select Forecast Horizon', min_value=1, max_value=min(365, len(df)), value=min(30, len(df)))
        model_name = "timegpt-1"
        if forecast_horizon > 12:
            model_name = "timegpt-1-long-horizon"

        if st.button('Run Forecast'):
            # Run models
            with st.spinner('Running Prophet...'):
                prophet_forecast = run_prophet(df, forecast_horizon)

            with st.spinner('Running TimeGPT...'):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                timegpt_forecast = loop.run_until_complete(run_timegpt(df, h=forecast_horizon, model=model_name))
                plot = st.session_state.timegpt.plot(df, timegpt_forecast, engine='plotly')

            with st.spinner('Running ARIMA...'):
                arima_forecast = run_arima(df, forecast_horizon)

            with st.spinner('Running SARIMA...'):
                sarima_forecast = run_sarima(df, forecast_horizon)

            with st.spinner('Running Exponential Smoothing...'):
                exp_smooth_forecast = run_exponential_smoothing(df, forecast_horizon)

            # Plot forecasts
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df['ds'], y=df['y'], mode='lines', name='Original Data'))
            fig.add_trace(go.Scatter(x=prophet_forecast['ds'][-forecast_horizon:], y=prophet_forecast['yhat'][-forecast_horizon:], mode='lines', name='Prophet'))
            fig.add_trace(go.Scatter(x=df['ds'].iloc[-forecast_horizon:], y=timegpt_forecast['TimeGPT'][-forecast_horizon:], mode='lines', name='TimeGPT'))
            fig.add_trace(go.Scatter(x=df['ds'].iloc[-forecast_horizon:], y=arima_forecast, mode='lines', name='ARIMA'))
            fig.add_trace(go.Scatter(x=df['ds'].iloc[-forecast_horizon:], y=sarima_forecast, mode='lines', name='SARIMA'))
            fig.add_trace(go.Scatter(x=df['ds'].iloc[-forecast_horizon:], y=exp_smooth_forecast, mode='lines', name='Exponential Smoothing'))
            
            fig.update_layout(title='Forecast Comparison', xaxis_title='Date', yaxis_title='Value')
            st.plotly_chart(fig)

            # Evaluate models
            actual = df['y'].iloc[-forecast_horizon:]
            prophet_eval = evaluate_model(actual, prophet_forecast['yhat'][-forecast_horizon:])
            timegpt_eval = evaluate_model(actual, timegpt_forecast['TimeGPT'][-forecast_horizon:])
            arima_eval = evaluate_model(actual, arima_forecast)
            sarima_eval = evaluate_model(actual, sarima_forecast)
            exp_smooth_eval = evaluate_model(actual, exp_smooth_forecast)

            # Determine the best model based on RMSE
            models = {
                'Prophet': prophet_eval,
                'TimeGPT': timegpt_eval,
                'ARIMA': arima_eval,
                'SARIMA': sarima_eval,
                'Exponential Smoothing': exp_smooth_eval
            }

            best_model = min(models, key=lambda x: models[x]['RMSE'])
            best_model_metrics = models[best_model]
            
            # Display evaluation metrics
            st.subheader('Evaluation Metrics')
            metrics_df = pd.DataFrame({
                'Prophet': prophet_eval,
                'TimeGPT': timegpt_eval,
                'ARIMA': arima_eval,
                'SARIMA': sarima_eval,
                'Exponential Smoothing': exp_smooth_eval
            }).T
            st.table(metrics_df)

            # Display best model
            st.subheader('Best Model')
            st.success(f"The best model is **{best_model}** with the following evaluation metrics:")
            st.write(f"- **RMSE**: {best_model_metrics['RMSE']:.2f}")
            st.write(f"- **MAE**: {best_model_metrics['MAE']:.2f}")
            st.write(f"- **MAPE**: {best_model_metrics['MAPE']:.2f}%")
            st.write(f"- **SMAPE**: {best_model_metrics['SMAPE']:.2f}%")
            st.write(f"- **MASE**: {best_model_metrics['MASE']:.2f}")

if __name__ == '__main__':
    main()
