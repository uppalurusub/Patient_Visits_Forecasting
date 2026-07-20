import {
    Card,
    CardContent,
    Chip,
    Divider,
    Grid,
    Stack,
    Typography
} from "@mui/material";

import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import AccessTimeIcon from "@mui/icons-material/AccessTime";

import type { RegressionPredictResponse } from "../../models/RegressionPredictResponse";

interface RegressionResultProps {

    result?: RegressionPredictResponse | null;

}

const RegressionResult = ({
    result
}: RegressionResultProps) => {

    if (!result) {

        return null;

    }

    return (

        <Card
            elevation={4}
            sx={{
                borderRadius: 3,
                mt: 3
            }}
        >

            <CardContent>

                <Stack
                    direction="row"
                    justifyContent="space-between"
                    alignItems="center"
                    mb={2}
                >

                    <Typography variant="h5">

                        Regression Prediction

                    </Typography>

                    <Chip
                        color="success"
                        label="Prediction Generated"
                    />

                </Stack>

                <Divider sx={{ mb: 3 }} />

                <Grid
                    container
                    spacing={3}
                >

                    <Grid size={{ xs: 12, md: 6 }}>

                        <Typography
                            variant="subtitle2"
                            color="text.secondary"
                        >

                            Predicted Value

                        </Typography>

                        <Stack
                            direction="row"
                            spacing={1}
                            alignItems="center"
                            mt={1}
                        >

                            <TrendingUpIcon
                                color="primary"
                                fontSize="large"
                            />

                            <Typography
                                variant="h3"
                                color="primary"
                                fontWeight="bold"
                            >

                                {result.prediction.toFixed(2)}

                            </Typography>

                        </Stack>

                    </Grid>

                    <Grid size={{ xs: 12, md: 6 }}>

                        <Typography
                            variant="subtitle2"
                            color="text.secondary"
                        >

                            Generated On

                        </Typography>

                        <Stack
                            direction="row"
                            spacing={1}
                            alignItems="center"
                            mt={1}
                        >

                            <AccessTimeIcon color="action" />

                            <Typography variant="body1">

                                {new Date().toLocaleString()}

                            </Typography>

                        </Stack>

                    </Grid>

                </Grid>

            </CardContent>

        </Card>

    );

};

export default RegressionResult;