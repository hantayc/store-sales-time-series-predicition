# Store Sales Time Series Forecasting

This repository contains a personal project for forecasting store-level sales using time series methods (SARIMAX) with exogenous regressors (holidays). The goal is to demonstrate end-to-end data preparation, modeling and evaluation. 

## Project Overview

**Dataset**: Daily sales data by store and product family from the Kaggle "Store Sales - Time Series Forecasting" competition. Including supplemental dataset, holidays, to indicate holiday date events. 

**Objective**: Build and evaluate one-step-ahead SARIMAX model to predict daily (and aggregated monthly) sales, incorporating holiday indicators as exogenous variables. 

**Key Steps**:

1. Data loading and cleaning 

2. Aggregation to daily and monthly totals 

3. Feature engineering (holiday flag, date parsing)

4. Model fitting with pmdarima.auto_arima and walk-forward forecasting 

5. Evaluation using RMSE, visualization of actuals vs. predictions 

# Monthly Total Sales Forecasing with SARIMA

## Data Description 

After basic EDA, we observe that there has been systematic growth in sales (steadily increasing year over year) for specific product families. However, for a learning experiment we do not consider modeling for specific product families due to having to create different models, using the SARIMAX method, for each product families. Further, we consider the high cardinatlity of the product family feature. 

We observe the following total monthly sales growth from 2013 to 2017 in the training data as shown below: 

![Total Sale Growth Training Data](/images/image.png)

Observing monthly sale data, we assume a seasonality of 12 periods, 12 months per year. Observing that in the beginning of the year we see a decline in total sales, after experiencing a spike typically towards the end of the year. 

## Model Build 

For this learning project, we use the **auto_arima()** function provided by the pmdarima package. 

I start with trimming the 2017 year from train_df, because after observing test_df csv file from the Kaggle data, I see that there are no total sales available as a feature in that file. Thus, we would need to leverage the training data set to predict n periods. For the purpose of this project, we use 2013 to 2016 values to predict total sales monetary value for 1/1/2017 through 7/1/2027. We exclude 08/1/2017 because after observing the data, we assume that the sale volume indicates that it's not a full complete month's worth of data. 

Code used to implement SARIMA model. 

```
warnings.simplefilter('ignore', FutureWarning)

sarima_model = auto_arima(
    train_subset['total_sales'],
    seasonal=True,
    m=7,
    start_p=1, start_q=1,
    max_p=5, max_q=5,           
    start_P=0, start_Q=0,
    max_P=1, max_Q=1,           # seasonal ar/ma orderes
    D=1,                        # seasonal difference 
    stepwise=True,              # stepwise algo 
    trace=True,
    error_action='ignore',
    suppress_warnings=True
)

print(sarima_model)

```

As mentioned previously, we use `auto_arima()` function to automatically select and fit the best SARIMAX model parameters by minimizing the Akaike Information Criterion. The AIC is a metric capturing model fit against model complexity. Lower the AIC indicates a model that explains the data well without overfitting `auto_arima()` uses AIC to rank candidate models and pick the one with the smallest AIC. Stepwise search we set to true to start from initial orders `(start_p, start_q, star_P, start_Q)` and after each iteration, we adjust a single parameter one at a time up or down. The stepwise arguemnt continues changing if it lowers the AIC; otherise, it reverts and tried another. This demostrates a greedy approach for finding near optiomal settings. 

After fitting the SARIMA model on the training data, we end with the following p,d,q,m,P,D,Q orders. 

`ARIMA(5,0,3)(0,1,1)[7]` 

The correct interpretation of this is the following for the non-seasonal order `(p,d,q) = (5,0,3)` using the last 5 lagged values and incorporate 3 lagged error terms. The seasonal order is `(P,D,Q) = (0,1,1)` at period 7 captures one seasonal difference and one seasonal MA term to model weekly seasonality. 

After running this model on the test dataset, we observe a RMSE (Root Mean Squared Error) of $238,847. 

![sarima: preds vs actuals](/images/sarima-timeseries.png)

## SARIMAX 

We run a SARIMAX model to capture the exogenous signals that is provided from holidays dataframe. Again, for a learning excercise we use the automated SARIMAC selection. The result of the SARIMAX `auto_arima` is `ARIMA(5,0,3)(0,1,1)[7] intercept`. We are working with daily data so we set seasonality for 7 periods, making it assume weekly seasonality. We interprete the ARIMA using the same understanding from above. The best fitting SARIMAX model is five lagged sales terms and three lagged errors (non-seasonal). 

We use walk-forward forecasting where rather than batch-forecast all of 2017, we predict the next 1 timeframe and then update the model with the actuals. This approach prevents look-ahead bias and more faithfully mirrors a production deployment. 

![sarima-plot](/images/sarimax-plot.png)

#### Intrepretations & Takeaways 

**Holiday Signal**: The binary holiday flag modestly nudges the forecast downwards on days like New Year's, but most of the forecast's accuracy comes from capturing the regular weekly ups and downs. 

**Weekly Seasonality**: Evan a single seasonal MA term at lag 7 yielded good improvement. Learning that mid-week and weekend sales follow different patterns. 

**Evaluation**: Adding the "X" part in SARIMAX provided value in terms of being able to bring down the RMSE to 158,588.00.