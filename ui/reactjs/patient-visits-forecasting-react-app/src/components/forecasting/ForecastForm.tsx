import { useState } from "react";

import {
    Box,
    Button,
    CircularProgress,
    Grid,
    Paper,
    Stack,
    TextField,
    Typography,
} from "@mui/material";

import forecastingApi from "../../api/forecastingApi";

import type { PredictRequest } from "../../models/PredictRequest";
import type { PredictionResponse } from "../../models/PredictionResponse";

interface ForecastFormProps {
    onPrediction: (response: PredictionResponse) => void;
}

const initialState: PredictRequest = {
    model_name: "AR",
    horizon: 7,
};

const ForecastForm = ({
    onPrediction,
}: ForecastFormProps) => {
    const [formData, setFormData] =
        useState<PredictRequest>(initialState);

    const [loading, setLoading] =
        useState(false);

    const [error, setError] =
        useState("");

    const handleHorizonChange = (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {
        setFormData((prev) => ({
            ...prev,
            horizon: Number(event.target.value),
        }));
    };

    const handleReset = () => {
        setFormData(initialState);
        setError("");
    };

    const handleSubmit = async (
        event: React.FormEvent<HTMLFormElement>
    ) => {
        event.preventDefault();

        if (formData.horizon < 1) {
            setError("Forecast horizon must be greater than zero.");
            return;
        }

        setLoading(true);
        setError("");

        try {
            const response =
                await forecastingApi.predict(formData);

            onPrediction(response);
        } catch (err: any) {
            setError(
                err.response?.data?.detail ??
                err.message ??
                "Forecast prediction failed."
            );
        } finally {
            setLoading(false);
        }
    };

    return (
        <Paper
            elevation={3}
            sx={{
                p: 4,
                borderRadius: 3,
            }}
        >
            <Typography
                variant="h5"
                gutterBottom
            >
                Time Series Forecast
            </Typography>

            <Typography
                variant="body2"
                color="text.secondary"
                sx={{ mb: 3 }}
            >
                Model: <strong>AR (Auto Regression)</strong>
            </Typography>

            <Box
                component="form"
                onSubmit={handleSubmit}
            >
                <Grid
                    container
                    spacing={3}
                >
                    <Grid size={{ xs: 12, md: 6 }}>
                        <TextField
                            fullWidth
                            label="Forecast Model"
                            value="AR (Auto Regression)"
                            InputProps={{
                                readOnly: true,
                            }}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>
                        <TextField
                            fullWidth
                            label="Forecast Horizon"
                            type="number"
                            value={formData.horizon}
                            onChange={handleHorizonChange}
                            inputProps={{
                                min: 1,
                            }}
                        />
                    </Grid>
                </Grid>

                {error && (
                    <Typography
                        color="error"
                        sx={{ mt: 2 }}
                    >
                        {error}
                    </Typography>
                )}

                <Stack
                    direction="row"
                    spacing={2}
                    sx={{ mt: 4 }}
                >
                    <Button
                        variant="contained"
                        type="submit"
                        disabled={loading}
                    >
                        {loading ? (
                            <CircularProgress
                                size={22}
                                color="inherit"
                            />
                        ) : (
                            "Generate Forecast"
                        )}
                    </Button>

                    <Button
                        variant="outlined"
                        onClick={handleReset}
                        disabled={loading}
                    >
                        Reset
                    </Button>
                </Stack>
            </Box>
        </Paper>
    );
};

export default ForecastForm;