import { CssBaseline, ThemeProvider, createTheme } from "@mui/material";

import AppRoutes from "./routes/AppRoutes";

const theme = createTheme({

    palette: {

        mode: "light",

        primary: {

            main: "#1976d2"

        },

        secondary: {

            main: "#2e7d32"

        },

        background: {

            default: "#f5f5f5",

            paper: "#ffffff"

        }

    },

    shape: {

        borderRadius: 10

    },

    typography: {

        fontFamily: [
            "Roboto",
            "Helvetica",
            "Arial",
            "sans-serif"
        ].join(","),

        h4: {

            fontWeight: 700

        },

        h5: {

            fontWeight: 600

        },

        h6: {

            fontWeight: 600

        }

    },

    components: {

        MuiPaper: {

            styleOverrides: {

                root: {

                    borderRadius: 10

                }

            }

        },

        MuiCard: {

            styleOverrides: {

                root: {

                    borderRadius: 12

                }

            }

        },

        MuiButton: {

            defaultProps: {

                variant: "contained"

            }

        }

    }

});

function App() {

    return (

        <ThemeProvider theme={theme}>

            <CssBaseline />

            <AppRoutes />

        </ThemeProvider>

    );

}

export default App;