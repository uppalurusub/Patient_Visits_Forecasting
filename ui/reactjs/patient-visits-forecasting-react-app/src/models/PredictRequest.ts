/**
 * Forecast Prediction Request
 */
export interface PredictRequest {

    /**
     * Forecasting model name.
     * Example:
     * AR
     * ARIMA
     * SARIMA
     * Prophet
     * LSTM
     */
    model_name: string;

    /**
     * Number of future periods to forecast.
     */
    horizon: number;

}