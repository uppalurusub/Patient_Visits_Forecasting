import {
    BrowserRouter,
    Routes,
    Route,
    Navigate,
} from "react-router-dom";

import PredictionPage from "../pages/PredictionPage";

export default function AppRoutes() {
    return (
        <BrowserRouter>
            <Routes>

                <Route
                    path="/"
                    element={<Navigate to="/forecast" replace />}
                />

                <Route
                    path="/forecast"
                    element={<PredictionPage />}
                />

            </Routes>
        </BrowserRouter>
    );
}