Stock Prediction is a hot evergreen topic. People want to be rich fast and no better way than trading in stock market.
Here is a rudimentary stock prediction model - which can be improved a lot, however, as a start it works great.
the data used for EDA and modelling was from APP. csv file which has the data for that particular stock from 2012 - 2019.

EDA- Basic check of how the data looks like, any patterns we get to see in the given data.

Modelling - Several features can be extracted from the stock data like moving averages, monthly average, weekly average etc. These could be fed to a model to improve its prediction. However, here the LSTM model was trained only on the 
closing price feature and a prediction of 30 days was made. The model was compared with other models using stats like MAE, RMSE, R-square value, AIC and BIC.
The LSTM model proved to be great and therefore it was saved using joblib.

Using streamlit app the model can be deployed. There are certain features of this app:
- The app asks for a csv or excel file with the stock data
- The column names should be checked in the given csv file
- The app asks for a time period for prediction 1-60 day max.
