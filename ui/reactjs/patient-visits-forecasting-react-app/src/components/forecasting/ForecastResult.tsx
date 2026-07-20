import {
    Card,
    CardContent,
    Chip,
    Divider,
    Grid,
    Paper,
    Stack,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Typography
} from "@mui/material";

import CheckCircleIcon from "@mui/icons-material/CheckCircle";
import TrendingUpIcon from "@mui/icons-material/TrendingUp";
import TimelineIcon from "@mui/icons-material/Timeline";

import type { PredictionResponse } from "../../models/PredictionResponse";

interface ForecastResultProps {

    result?: PredictionResponse | null;

}

const ForecastResult = ({
    result
}: ForecastResultProps) => {

    if (!result) {

        return null;

    }

    const values = result.prediction;

    const minimum = Math.min(...values);

    const maximum = Math.max(...values);

    const average =
        values.reduce((a, b) => a + b, 0) / values.length;

    return (

        <Card
            elevation={4}
            sx={{
                mt: 3,
                borderRadius: 3
            }}
        >

            <CardContent>

                <Stack
                    direction="row"
                    justifyContent="space-between"
                    alignItems="center"
                >

                    <Typography variant="h5">

                        Forecast Results

                    </Typography>

                    <Chip
                        color="success"
                        icon={<CheckCircleIcon />}
                        label="Forecast Generated"
                    />

                </Stack>

                <Divider sx={{ my: 3 }} />

                <Grid
                    container
                    spacing={2}
                >

                    <Grid size={{ xs: 12, md: 4 }}>

                        <Paper
                            elevation={2}
                            sx={{
                                p: 2,
                                textAlign: "center"
                            }}
                        >

                            <Typography
                                variant="subtitle2"
                                color="text.secondary"
                            >

                                Model

                            </Typography>

                            <Typography variant="h5">

                                {result.model_name}

                            </Typography>

                        </Paper>

                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>

                        <Paper
                            elevation={2}
                            sx={{
                                p: 2,
                                textAlign: "center"
                            }}
                        >

                            <Typography
                                variant="subtitle2"
                                color="text.secondary"
                            >

                                Forecast Points

                            </Typography>

                            <Typography variant="h5">

                                {values.length}

                            </Typography>

                        </Paper>

                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>

                        <Paper
                            elevation={2}
                            sx={{
                                p: 2,
                                textAlign: "center"
                            }}
                        >

                            <Typography
                                variant="subtitle2"
                                color="text.secondary"
                            >

                                Status

                            </Typography>

                            <Typography
                                variant="h5"
                                color="success.main"
                            >

                                Success

                            </Typography>

                        </Paper>

                    </Grid>

                </Grid>

                <Divider sx={{ my: 3 }} />

                <Typography
                    variant="h6"
                    gutterBottom
                >

                    <TimelineIcon
                        sx={{
                            mr: 1,
                            verticalAlign: "middle"
                        }}
                    />

                    Forecast Values

                </Typography>

                <TableContainer
                    component={Paper}
                    elevation={1}
                >

                    <Table>

                        <TableHead>

                            <TableRow>

                                <TableCell>

                                    Day

                                </TableCell>

                                <TableCell align="right">

                                    Forecast

                                </TableCell>

                            </TableRow>

                        </TableHead>

                        <TableBody>

                            {

                                values.map((value, index) => (

                                    <TableRow
                                        key={index}
                                    >

                                        <TableCell>

                                            Day {index + 1}

                                        </TableCell>

                                        <TableCell align="right">

                                            {value.toFixed(2)}

                                        </TableCell>

                                    </TableRow>

                                ))

                            }

                        </TableBody>

                    </Table>

                </TableContainer>

                <Divider sx={{ my: 3 }} />

                <Typography
                    variant="h6"
                    gutterBottom
                >

                    <TrendingUpIcon
                        sx={{
                            mr: 1,
                            verticalAlign: "middle"
                        }}
                    />

                    Forecast Statistics

                </Typography>

                <Grid
                    container
                    spacing={2}
                >

                    <Grid size={{ xs: 12, md: 4 }}>

                        <Paper
                            sx={{
                                p: 2,
                                textAlign: "center"
                            }}
                        >

                            <Typography
                                color="text.secondary"
                            >

                                Minimum

                            </Typography>

                            <Typography
                                variant="h5"
                            >

                                {minimum.toFixed(2)}

                            </Typography>

                        </Paper>

                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>

                        <Paper
                            sx={{
                                p: 2,
                                textAlign: "center"
                            }}
                        >

                            <Typography
                                color="text.secondary"
                            >

                                Average

                            </Typography>

                            <Typography
                                variant="h5"
                            >

                                {average.toFixed(2)}

                            </Typography>

                        </Paper>

                    </Grid>

                    <Grid size={{ xs: 12, md: 4 }}>

                        <Paper
                            sx={{
                                p: 2,
                                textAlign: "center"
                            }}
                        >

                            <Typography
                                color="text.secondary"
                            >

                                Maximum

                            </Typography>

                            <Typography
                                variant="h5"
                            >

                                {maximum.toFixed(2)}

                            </Typography>

                        </Paper>

                    </Grid>

                </Grid>

                <Divider sx={{ my: 3 }} />

                <Typography
                    variant="body2"
                    color="text.secondary"
                >

                    {result.message}

                </Typography>

            </CardContent>

        </Card>

    );

};

export default ForecastResult;