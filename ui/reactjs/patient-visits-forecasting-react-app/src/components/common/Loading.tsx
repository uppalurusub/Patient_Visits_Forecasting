import {
    Backdrop,
    Box,
    CircularProgress,
    Stack,
    Typography
} from "@mui/material";

interface LoadingProps {

    /**
     * Loading message
     */
    message?: string;

    /**
     * Spinner size
     */
    size?: number;

    /**
     * Display as fullscreen backdrop
     */
    fullscreen?: boolean;

}

const Loading = ({
    message = "Loading...",
    size = 50,
    fullscreen = false
}: LoadingProps) => {

    const content = (

        <Stack
            spacing={2}
            alignItems="center"
            justifyContent="center"
        >

            <CircularProgress
                size={size}
            />

            <Typography
                variant="body1"
                color="text.secondary"
            >

                {message}

            </Typography>

        </Stack>

    );

    if (fullscreen) {

        return (

            <Backdrop
                open
                sx={{
                    color: "#fff",
                    zIndex: (theme) => theme.zIndex.drawer + 1
                }}
            >

                {content}

            </Backdrop>

        );

    }

    return (

        <Box
            display="flex"
            justifyContent="center"
            alignItems="center"
            minHeight={250}
            width="100%"
        >

            {content}

        </Box>

    );

};

export default Loading;