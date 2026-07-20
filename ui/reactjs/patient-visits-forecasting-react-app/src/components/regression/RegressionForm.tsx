import { useState } from "react";
import {
    Box,
    Button,
    CircularProgress,
    Grid,
    Paper,
    Stack,
    TextField,
    Typography
} from "@mui/material";

import forecastingApi from "../../api/forecastingApi";

import type { RegressionPredictRequest } from "../../models/RegressionPredictRequest";
import type { RegressionPredictResponse } from "../../models/RegressionPredictResponse";

interface RegressionFormProps {
    onPrediction: (response: RegressionPredictResponse) => void;
}

<Grid size={{ xs: 12 }}>

    <TextField

        fullWidth

        label="Regression Model"

        value="ElasticNet Regression"

        InputProps={{
            readOnly: true
        }}

    />

</Grid>

const initialState: RegressionPredictRequest = {

    model_name: "ELASTICNET",
    patient_count: 120,
    admissions: 32,
    discharges: 30,
    occupancy: 71.5,
    year: 2024,
    month: 6,
    day: 14,
    dayofweek: 4,
    quarter: 2,
    weekofyear: 24

};

const RegressionForm = ({
    onPrediction,
}: RegressionFormProps) => {

    const [formData, setFormData] =
        useState<RegressionPredictRequest>(initialState);

    const [loading, setLoading] = useState(false);

    const [error, setError] = useState("");

    const handleChange = (
        event: React.ChangeEvent<HTMLInputElement>
    ) => {

        const { name, value } = event.target;

        setFormData((prev) => ({
            ...prev,
            [name]: Number(value),
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

        setLoading(true);

        setError("");

        try {

            const response =
                await forecastingApi.regressionPredict(formData);

            onPrediction(response);

        } catch (err: any) {

            setError(
                err.response?.data?.detail ??
                err.message ??
                "Prediction failed."
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
                borderRadius: 3
            }}
        >

            <Typography
                variant="h5"
                gutterBottom
            >

                Regression Prediction

            </Typography>

            <Box
                component="form"
                onSubmit={handleSubmit}
            >

                <Grid
                    container
                    spacing={2}
                >

                    <Grid size={{ xs: 12, md: 6 }}>
                        <TextField
                            fullWidth
                            label="Patient Count"
                            name="patient_count"
                            type="number"
                            value={formData.patient_count}
                            onChange={handleChange}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>
                        <TextField
                            fullWidth
                            label="Admissions"
                            name="admissions"
                            type="number"
                            value={formData.admissions}
                            onChange={handleChange}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>
                        <TextField
                            fullWidth
                            label="Discharges"
                            name="discharges"
                            type="number"
                            value={formData.discharges}
                            onChange={handleChange}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>
                        <TextField
                            fullWidth
                            label="Occupancy (%)"
                            name="occupancy"
                            type="number"
                            value={formData.occupancy}
                            onChange={handleChange}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>
                        <TextField
                            fullWidth
                            label="Year"
                            name="year"
                            type="number"
                            value={formData.year}
                            onChange={handleChange}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>
                        <TextField
                            fullWidth
                            label="Month"
                            name="month"
                            type="number"
                            value={formData.month}
                            onChange={handleChange}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>
                        <TextField
                            fullWidth
                            label="Day"
                            name="day"
                            type="number"
                            value={formData.day}
                            onChange={handleChange}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>
                        <TextField
                            fullWidth
                            label="Day Of Week"
                            name="dayofweek"
                            type="number"
                            value={formData.dayofweek}
                            onChange={handleChange}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>
                        <TextField
                            fullWidth
                            label="Quarter"
                            name="quarter"
                            type="number"
                            value={formData.quarter}
                            onChange={handleChange}
                        />
                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>
                        <TextField
                            fullWidth
                            label="Week Of Year"
                            name="weekofyear"
                            type="number"
                            value={formData.weekofyear}
                            onChange={handleChange}
                        />
                    </Grid>

                </Grid>

                {

                    error && (

                        <Typography
                            color="error"
                            sx={{ mt: 2 }}
                        >

                            {error}

                        </Typography>

                    )

                }

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

                        {

                            loading
                                ? <CircularProgress size={22} />
                                : "Predict"

                        }

                    </Button>

                    <Button
                        variant="outlined"
                        onClick={handleReset}
                    >

                        Reset

                    </Button>

                </Stack>

            </Box>

        </Paper>

    );

};

export default RegressionForm;