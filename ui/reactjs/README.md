# 📈 Healthcare Forecasting Frontend

A modern **React + TypeScript** web application for performing **Time Series Forecasting** and **Regression Predictions** through a FastAPI backend.

The application provides an intuitive Material UI interface where users can submit forecasting requests, view prediction results, and visualize model outputs.

---

# 🚀 Features

- Modern React 19 + TypeScript
- Built with Vite
- Material UI (MUI) Design
- Axios API Integration
- React Router Navigation
- Time Series Forecasting
- Regression Prediction
- Markdown-friendly Prediction Results
- Responsive Layout
- Clean Component Architecture
- Strongly Typed Models

---

# 🏗 Project Architecture

```
src
│
├── api
│   ├── apiClient.ts
│   └── forecastingApi.ts
│
├── assets
│
├── components
│   ├── forecasting
│   │     ├── ForecastForm.tsx
│   │     └── ForecastResult.tsx
│   │
│   └── regression
│         ├── RegressionForm.tsx
│         └── RegressionResult.tsx
│
├── models
│   ├── PredictRequest.ts
│   ├── PredictionRequest.ts
│   ├── PredictionResponse.ts
│   ├── RegressionPredictRequest.ts
│   └── RegressionPredictResponse.ts
│
├── pages
│   ├── PredictionPage.tsx
│   ├── ForecastPage.tsx
│   ├── RegressionPrediction.tsx
│   └── TimeSeriesPrediction.tsx
│
├── routes
│   └── AppRoutes.tsx
│
├── utils
│   └── numberFormatter.ts
│
├── App.tsx
├── main.tsx
└── index.css
```

---

# 🛠 Technology Stack

| Technology | Version |
|------------|---------|
| React | 19 |
| TypeScript | Latest |
| Vite | Latest |
| Material UI | Latest |
| Axios | Latest |
| React Router DOM | Latest |

---

# 📦 Installation

Clone the repository

```bash
git clone <repository-url>
```

Navigate to project

```bash
cd forecasting-frontend
```

Install dependencies

```bash
npm install
```

---

# ▶ Running the Application

Start development server

```bash
npm run dev
```

Application runs at

```
http://localhost:5173
```

---

# 🏗 Build Production

```bash
npm run build
```

Preview production build

```bash
npm run preview
```

---

# 🔗 Backend Configuration

The application communicates with a FastAPI backend.

Current API configuration:

```typescript
baseURL: http://localhost:8000/api/v1
```

Located in

```
src/api/apiClient.ts
```

Example

```typescript
export default axios.create({
    baseURL: "http://localhost:8000/api/v1",
    headers: {
        "Content-Type": "application/json"
    }
});
```

---

# API Endpoints

## Time Series Prediction

```
POST /forecast/predict
```

## Regression Prediction

```
POST /forecast/regression-predict
```

---

# Time Series Forecast Workflow

```
User
   │
   ▼
Forecast Form
   │
   ▼
Axios
   │
   ▼
FastAPI
   │
   ▼
Forecast Model
   │
   ▼
Prediction Response
   │
   ▼
Forecast Result Component
```

---

# Regression Workflow

```
User
   │
   ▼
Regression Form
   │
   ▼
Axios
   │
   ▼
FastAPI
   │
   ▼
Regression Model
   │
   ▼
Prediction Response
   │
   ▼
Regression Result Component
```

---

# Routing

The application currently exposes the following route.

| Route | Description |
|--------|-------------|
| / | Redirects to Forecast |
| /forecast | Prediction Dashboard |

---

# Application Modules

## Time Series Forecast

Allows users to submit historical data and generate future forecasts.

Features

- Forecast Horizon
- Future Predictions
- Confidence Intervals
- Model Summary
- Forecast Metrics

---

## Regression Prediction

Predicts a numerical outcome based on input features.

Features

- Dynamic Input Form
- Regression Prediction
- Confidence Score
- Explanation Output

---

# API Layer

The API layer is responsible for communication with the backend.

```
forecastingApi.ts
```

Functions

```typescript
predict()

regressionPredict()
```

---

# Component Architecture

```
PredictionPage
│
├── ForecastForm
│       │
│       └── ForecastResult
│
└── RegressionForm
        │
        └── RegressionResult
```

---

# Models

The project uses strongly typed request and response models.

Examples

```
PredictRequest

PredictionResponse

RegressionPredictRequest

RegressionPredictResponse
```

---

# UI Components

- Material UI Theme
- Cards
- Forms
- Select Boxes
- Buttons
- Typography
- Responsive Containers

---

# Error Handling

The application supports

- API validation errors
- Network failures
- Backend exceptions
- Invalid user input
- Empty responses

---

# Future Enhancements

- Authentication
- User Management
- Forecast Charts
- Download Predictions (CSV)
- PDF Reports
- Model Selection
- Forecast History
- Dashboard Analytics
- Theme Switcher
- Dark Mode

---

# Development Commands

Install packages

```bash
npm install
```

Start

```bash
npm run dev
```

Build

```bash
npm run build
```

Preview

```bash
npm run preview
```

Lint

```bash
npm run lint
```

---

# Best Practices Followed

- Component-Based Architecture
- Separation of Concerns
- Type Safety
- Reusable Components
- Modular API Layer
- Strong Typing
- Responsive Design
- Maintainable Folder Structure

---

# License

MIT License

---

# Author

Developed using

- React
- TypeScript
- Material UI
- Axios
- FastAPI
- Vite

```

### Project Summary

This application serves as a modern frontend client for forecasting services, enabling users to perform both **time series forecasting** and **regression-based predictions** through a clean, responsive interface backed by a FastAPI API. It follows a modular architecture with reusable components, typed models, and a dedicated API layer, making it suitable for healthcare analytics, demand forecasting, patient volume prediction, and other machine learning inference scenarios.