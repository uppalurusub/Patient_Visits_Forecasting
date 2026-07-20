import { Alert, AlertTitle, Button, Collapse } from "@mui/material";
import { useState } from "react";
import type { AlertColor } from "@mui/material";

interface ErrorAlertProps {

    /**
     * Alert severity
     */
    severity?: AlertColor;

    /**
     * Alert title
     */
    title?: string;

    /**
     * Alert message
     */
    message: string;

    /**
     * Show close icon
     */
    closable?: boolean;

    /**
     * Retry callback
     */
    onRetry?: () => void;

}

const ErrorAlert = ({

    severity = "error",

    title,

    message,

    closable = true,

    onRetry

}: ErrorAlertProps) => {

    const [open, setOpen] = useState(true);

    if (!message) {

        return null;

    }

    return (

        <Collapse in={open}>

            <Alert

                severity={severity}

                sx={{

                    mt: 2,

                    mb: 2,

                    borderRadius: 2

                }}

                onClose={

                    closable

                        ? () => setOpen(false)

                        : undefined

                }

                action={

                    onRetry && (

                        <Button

                            color="inherit"

                            size="small"

                            onClick={onRetry}

                        >

                            Retry

                        </Button>

                    )

                }

            >

                {

                    title && (

                        <AlertTitle>

                            {title}

                        </AlertTitle>

                    )

                }

                {message}

            </Alert>

        </Collapse>

    );

};

export default ErrorAlert;