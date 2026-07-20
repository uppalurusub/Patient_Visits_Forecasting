import { useState } from "react";

import {
  Box,
  FormControl,
  InputLabel,
  MenuItem,
  Select,
} from "@mui/material";

import ForecastForm from "../components/forecasting/ForecastForm";
import RegressionForm from "../components/regression/RegressionForm";

import type { PredictionResponse } from "../models/PredictionResponse";
import type { RegressionPredictResponse } from "../models/RegressionPredictResponse";

import ForecastResult from "../components/forecasting/ForecastResult";
import RegressionResult from "../components/regression/RegressionResult";

export default function PredictionPage() {
  const [type, setType] = useState<"timeseries" | "regression">(
    "timeseries"
  );

  const [forecastResult, setForecastResult] =
    useState<PredictionResponse | null>(null);

  const [regressionResult, setRegressionResult] =
    useState<RegressionPredictResponse | null>(null);

  return (
    <Box p={3}>
      <FormControl fullWidth sx={{ mb: 3 }}>
        <InputLabel id="prediction-type-label">
          Prediction Type
        </InputLabel>

        <Select
          labelId="prediction-type-label"
          value={type}
          label="Prediction Type"
          onChange={(e) =>
            setType(e.target.value as "timeseries" | "regression")
          }
        >
          <MenuItem value="timeseries">
            Time Series Forecast
          </MenuItem>

          <MenuItem value="regression">
            Regression Prediction
          </MenuItem>
        </Select>
      </FormControl>

      {type === "timeseries" ? (
        <>
          <ForecastForm
            onPrediction={(response) => {
              setForecastResult(response);
            }}
          />

          {forecastResult && (
            <ForecastResult 
                
                
                    result={forecastResult}
                
            
            />
          )}
        </>
      ) : (
        <>
          <RegressionForm
            onPrediction={(response) => {
              setRegressionResult(response);
            }}
          />

          {regressionResult && (
            <RegressionResult 
                result={regressionResult} 
            
            />
          )}
        </>
      )}
    </Box>
  );
}