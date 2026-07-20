import { useState } from "react";

import {
    Box,
    Container
} from "@mui/material";

import TimelineIcon from "@mui/icons-material/Timeline";

import PageHeader from "../components/common/PageHeader";
import Loading from "../components/common/Loading";
import ErrorAlert from "../components/common/ErrorAlert";

import ForecastForm from "../components/forecasting/ForecastForm";
import ForecastResult from "../components/forecasting/ForecastResult";

import type { PredictionResponse } from "../models/PredictionResponse";

const TimeSeriesPrediction = () => {

    const [result, setResult] =
        useState<PredictionResponse | null>(null);

    const [loading, setLoading] =
        useState(false);

    const [error, setError] =
        useState("");

    const handlePrediction = (
        response: PredictionResponse
    ) => {

        setLoading(true);

        try {

            setResult(response);

            setError("");

        }

        catch {

            setError("Unable to display forecast.");

        }

        finally {

            setLoading(false);

        }

    };

    return (

        <Container
            maxWidth="xl"
            sx={{
                mt: 4,
                mb: 4
            }}
        >

            <PageHeader

                title="Time Series Forecasting"

                subtitle="Generate future forecasts using statistical and Machine Learning models."

                icon={
                    <TimelineIcon
                        color="primary"
                        fontSize="large"
                    />
                }

                breadcrumbs={[
                    {
                        label: "Home",
                        href: "/"
                    },
                    {
                        label: "Forecasting"
                    }
                ]}

            />

            {

                error && (

                    <ErrorAlert

                        severity="error"

                        title="Forecast Error"

                        message={error}

                    />

                )

            }

            <Box mt={3}>

                <ForecastForm

                    onPrediction={handlePrediction}

                />

            </Box>

            {

                loading

                    ?

                    (

                        <Loading

                            message="Generating Forecast..."

                        />

                    )

                    :

                    (

                        result && (

                            <ForecastResult

                                result={result}

                            />

                        )

                    )

            }

        </Container>

    );

};

export default TimeSeriesPrediction;