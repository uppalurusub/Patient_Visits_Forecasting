import apiClient from "./apiClient";

import type { PredictRequest } from "../models/PredictRequest";
import type { PredictionResponse } from "../models/PredictionResponse";

import type { RegressionPredictRequest } from "../models/RegressionPredictRequest";
import type { RegressionPredictResponse } from "../models/RegressionPredictResponse";

/**
 * Forecasting API Service
 */
class ForecastingApi {

    /**
     * Time Series Prediction
     */
    async predict(
        request: PredictRequest
    ): Promise<PredictionResponse> {

        const response = await apiClient.post<PredictionResponse>(
            "/forecast/predict",
            request
        );

        return response.data;
    }

    /**
     * Regression Prediction
     */
    async regressionPredict(
        request: RegressionPredictRequest
    ): Promise<RegressionPredictResponse> {

        const response = await apiClient.post<RegressionPredictResponse>(
            "/forecast/regression-predict",
            request
        );

        return response.data;
    }

}

export default new ForecastingApi();