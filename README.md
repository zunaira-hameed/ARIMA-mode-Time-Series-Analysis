# 📈 Ethereum (ETH/USDT) Price Forecasting using ARIMA

This project focuses on forecasting the price of **Ethereum (ETH/USDT)** using **ARIMA** (AutoRegressive Integrated Moving Average) time series model. It includes end-to-end data processing, analysis, model building, evaluation, and forecasting.

## 📌 Project Objective

To analyze historical Ethereum price data and develop a statistical model to accurately **forecast future prices**, helping in financial trend analysis and investment insights.

---

## 🧰 Tools and Technologies

- **Programming Language**: Python  
- **Libraries**:
  - `pandas`
  - `numpy`
  - `matplotlib`, `seaborn`
  - `statsmodels`
  - `sklearn`
  - `pmdarima` *(optional for auto ARIMA)*

---

## 📊 Workflow

### 1. Data Collection
- Imported Ethereum price data from a CSV file (e.g., `ETH_USDT.csv`).

### 2. Data Preprocessing
- Converted `date` column to `datetime`.
- Set it as index and removed nulls or invalid values.

### 3. Exploratory Data Analysis (EDA)
- Plotted price trends
- Visualized moving averages
- Checked for trends, seasonality, and volatility

### 4. Stationarity Testing
- Applied **Augmented Dickey-Fuller (ADF)** Test
- Differenced the data to make it stationary if required

### 5. ACF & PACF Analysis
- Used **AutoCorrelation Function (ACF)** and **Partial ACF** plots to determine optimal ARIMA parameters (p, d, q)

### 6. ARIMA Model Building
- Trained the **ARIMA(p,d,q)** model — best configuration found: **ARIMA(5,1,1)**

### 7. Model Evaluation
- Evaluated using:
  - **Root Mean Square Error (RMSE)**
  - **Mean Absolute Percentage Error (MAPE)**

### 8. Forecasting
- Forecasted the next **30 days of Ethereum prices**
- Visualized forecasts with confidence intervals
