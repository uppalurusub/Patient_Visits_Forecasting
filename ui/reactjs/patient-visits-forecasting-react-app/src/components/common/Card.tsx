import { ReactNode } from "react";

import {
    Card as MuiCard,
    CardContent,
    CardHeader,
    CircularProgress,
    Divider,
    IconButton,
    Stack,
    Typography
} from "@mui/material";

interface CardProps {

    title: string;

    subtitle?: string;

    icon?: ReactNode;

    action?: ReactNode;

    children: ReactNode;

    loading?: boolean;

    elevation?: number;

}

const Card = ({

    title,

    subtitle,

    icon,

    action,

    children,

    loading = false,

    elevation = 3

}: CardProps) => {

    return (

        <MuiCard

            elevation={elevation}

            sx={{

                borderRadius: 3,

                height: "100%"

            }}

        >

            <CardHeader

                avatar={icon}

                title={

                    <Typography

                        variant="h6"

                        fontWeight={600}

                    >

                        {title}

                    </Typography>

                }

                subheader={subtitle}

                action={

                    action && (

                        <IconButton>

                            {action}

                        </IconButton>

                    )

                }

            />

            <Divider />

            <CardContent>

                {

                    loading

                    ?

                    (

                        <Stack

                            alignItems="center"

                            justifyContent="center"

                            spacing={2}

                            sx={{

                                minHeight: 180

                            }}

                        >

                            <CircularProgress />

                            <Typography>

                                Loading...

                            </Typography>

                        </Stack>

                    )

                    :

                    children

                }

            </CardContent>

        </MuiCard>

    );

};

export default Card;