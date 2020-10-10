# EECS 731 - Project 5 (Forecasting)
### Author: Jace Kline

## Description
1. Set up a data science project structure in a new git repository in your GitHub account
2. Download the product demand data set from https://www.kaggle.com/felixzhao/productdemandforecasting
3. Load the data set into panda data frames
4. Formulate one or two ideas on how feature engineering would help the data set to establish additional value using exploratory data analysis
5. Build one or more forecasting models to determine the demand for a particular product using the other columns as features
6. Document your process and results
7. Commit your notebook, source code, visualizations and other supporting files to the git repository in GitHub

## Summary
In this project we focused on building models for time series forecasting for particular products. After extensive feature engineering and data exploration, we chose three products from the original dataset to utilize in our time series models. We decided to use the ARMA and Hidden Markov models to fit our data. We concluded that the Hidden Markov model was a better fit for the chosen datasets. See the full report [here.](./notebooks/forecast.md)
