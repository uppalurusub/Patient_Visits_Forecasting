import { useState } from "react";

import {
    Box,
    Container
} from "@mui/material";

import PsychologyIcon from "@mui/icons-material/Psychology";

import PageHeader from "../components/common/PageHeader";
import Loading from "../components/common/Loading";
import ErrorAlert from "../components/common/ErrorAlert";

import RegressionForm from "../components/regression/RegressionForm";
import RegressionResult from "../components/regression/RegressionResult";

import type { RegressionPredictResponse } from "../models/RegressionPredictResponse";

const RegressionPrediction = () => {

    const [result, setResult] =
        useState<RegressionPredictResponse | null>(null);

    const [loading, setLoading] =
        useState(false);

    const [error, setError] =
        useState("");

    const handlePrediction = (
        response: RegressionPredictResponse
    ) => {

        setLoading(true);

        try {

            setResult(response);

            setError("");

        }

        catch {

            setError("Unable to display prediction.");

        }

        finally {

            setLoading(false);

        }

    };

    return (

        <Container
            maxWidth="xl"
            sx={{ mt: 4, mb: 4 }}
        >

            <PageHeader

                title="Regression Prediction"

                subtitle="Predict values using the trained Regression Model."

                icon={
                    <PsychologyIcon
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
                        label: "Regression"
                    }

                ]}

            />

            {

                error && (

                    <ErrorAlert

                        severity="error"

                        title="Prediction Error"

                        message={error}

                    />

                )

            }

            <Box mt={3}>

                <RegressionForm

                    onPrediction={handlePrediction}

                />

            </Box>

            {

                loading

                    ?

                    (

                        <Loading

                            message="Generating Prediction..."

                        />

                    )

                    :

                    (

                        result && (

                            <RegressionResult

                                result={result}

                            />

                        )

                    )

            }

        </Container>

    );

};

export default RegressionPrediction;