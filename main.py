import pandas as pd
import numpy as np
import datetime
import os
import math
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, Dropout, LSTM

from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from prophet import Prophet

from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from pandas.plotting import register_matplotlib_converters

#%%
data = pd.read_csv('D:/Files/Datasets/cryptocurrency/main/2023-07-09_BTC-USD_3mo_5m.csv')
datac = data.copy()
datac = datac.drop('Symbol', axis=1)

print('Null Values:', datac.isnull().values.sum())
print('NA values:', datac.isnull().values.any())
print('Starting Date',datac.iloc[0][0])
print('Ending Date',datac.iloc[-1][0])

datac['Datetime'] = pd.to_datetime(datac['Datetime'], format='%Y-%m-%d')
closedf = datac[['Close']]
print("Shape of close dataframe:", closedf.shape)

scaler=MinMaxScaler(feature_range=(0,1))
closedf=scaler.fit_transform(np.array(closedf).reshape(-1,1))

training_size=int(len(closedf)*0.95)
test_size=len(closedf)-training_size
train_data,test_data=closedf[0:training_size,:],closedf[training_size:,:]
print("train_data: ", train_data.shape)
print("test_data: ", test_data.shape)

# convert an array of values into a dataset matrix
def create_dataset(dataset, time_step=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 20

X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

print("X_train: ", X_train.shape)
print("y_train: ", y_train.shape)
print("X_test: ", X_test.shape)
print("y_test", y_test.shape)

# reshape input to be [samples, time steps, features] which is required for LSTM
X_train =X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)

print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)

#%% ARIMA
closedf_arima = datac[['Datetime', 'Close']]
result = adfuller(closedf_arima['Close'])
labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
for value,label in zip(result,labels):
    print(label+' : '+str(value) )

if result[1] <= 0.05:
    print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
else:
    print("weak evidence against null hypothesis,indicating it is non-stationary ")

closedf_arima['Sales First Difference'] = closedf_arima['Sales'] - closedf_arima['Sales'].shift(1)
closedf_arima['Seasonal First Difference']=closedf_arima['Sales']-closedf_arima['Sales'].shift(12)
closedf_arima.head()

'''
closedf_arima = datac[['Datetime', 'Close']]
#closedf_arima.index = closedf_arima.index.to_period('M')
# fit model
model = ARIMA(closedf_arima['Close'], order=(10,1,2))
model_fit = model.fit()
# summary of fit model
print(model_fit.summary())
# line plot of residuals
from pandas import DataFrame
residuals = DataFrame(model_fit.resid)
residuals.plot()
plt.show()
# density plot of residuals
residuals.plot(kind='kde')
plt.show()
# summary stats of residuals
print(residuals.describe())
from sklearn.metrics import mean_squared_error
from math import sqrt
# split into train and test sets
X = closedf_arima['Close'].values
size = int(len(X) * 0.9)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,0))
	model_fit = model.fit()
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))
# evaluate forecasts
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
# plot forecasts against actual outcomes
plt.figure()
plt.plot(test)
plt.plot(predictions, color='red')
plt.show()
'''
'''
closedf_arima = datac[['Datetime', 'Close']]
closedf_arima.set_index('Datetime',inplace=True)

test_result = adfuller(closedf_arima['Close'])
def adfuller_test(sales):
    result=adfuller(sales)
    labels = ['ADF Test Statistic','p-value','#Lags Used','Number of Observations']
    for value,label in zip(result,labels):
        print(label+' : '+str(value) )

    if result[1] <= 0.05:
        print("strong evidence against the null hypothesis(Ho), reject the null hypothesis. Data is stationary")
    else:
        print("weak evidence against null hypothesis,indicating it is non-stationary ")

adfuller_test(closedf_arima['Close'])
closedf_arima['Sales First Difference'] = closedf_arima['Close'] - closedf_arima['Close'].shift(1)
closedf_arima['Hour First Difference']=closedf_arima['Close']-closedf_arima['Close'].shift(12)
closedf_arima.head()

#%% Again testing if data is stationary
adfuller_test(closedf_arima['Hour First Difference'].dropna())
closedf_arima['Hour First Difference'].plot()
#%%
from pandas.plotting import autocorrelation_plot
autocorrelation_plot(closedf_arima['Close'])
plt.show()
#%%
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
import statsmodels.api as sm
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(closedf_arima['Hour First Difference'].dropna(),lags=40,ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(closedf_arima['Hour First Difference'].dropna(),lags=40,ax=ax2)

#%% For non-seasonal data
#p=1, d=1, q=0 or 1

from statsmodels.tsa.arima.model import ARIMA
model=ARIMA(closedf_arima['Close'],order=(1,1,1))
model_fit=model.fit()
model_fit.summary()
#%%
closedf_arima['forecast']=model_fit.predict(start=16028,end=16628,dynamic=True)
closedf_arima[['Close','forecast']].plot(figsize=(12,8))
#%%
import statsmodels.api as sm
model=sm.tsa.statespace.SARIMAX(closedf_arima['Close'],order=(1, 1, 1),seasonal_order=(1,1,1,12))
results=model.fit()
closedf_arima['forecast']=results.predict(start=16028,end=16628,dynamic=True)
#%%
closedf_arima[['Close','forecast']].plot(figsize=(12,8))

#%%
from pandas.tseries.offsets import DateOffset
future_dates=[closedf_arima.index[-1]+ DateOffset(months=x)for x in range(0,24)]
future_datest_df=pd.DataFrame(index=future_dates[1:],columns=closedf_arima.columns)

future_datest_df.tail()

future_df=pd.concat([closedf_arima,future_datest_df])

future_df['forecast'] = results.predict(start = 16629, end = 18000, dynamic= True)
future_df[['Close', 'forecast']].plot(figsize=(12, 8))
'''
'''
register_matplotlib_converters()
closedf_arima = datac[['Datetime', 'Close']]
closedf_arima = closedf_arima.set_index(['Datetime'])
rolling_mean = closedf_arima.rolling(window = 12).mean()
rolling_std = closedf_arima.rolling(window = 12).std()
plt.figure()
plt.plot(closedf_arima, color = 'blue', label = 'Original')
plt.plot(rolling_mean, color = 'red', label = 'Rolling Mean')
plt.legend(loc = 'best')
plt.title('Rolling Mean')
plt.show()

plt.figure()
plt.plot(rolling_std, color = 'black', label = 'Rolling Std')
plt.legend(loc = 'best')
plt.title('Rolling Standard Deviation')
plt.show()

result = adfuller(closedf_arima['Close'])
print('ADF Statistic: {}'.format(result[0]))
print('p-value: {}'.format(result[1]))
print('Critical Values:')
for key, value in result[4].items():
    print('\t{}: {}'.format(key, value))

plt.figure()
closedf_arima_log = np.log(closedf_arima)
plt.plot(closedf_arima_log)
plt.show()

def get_stationarity(timeseries):
    
    # rolling statistics
    rolling_mean = timeseries.rolling(window=12).mean()
    rolling_std = timeseries.rolling(window=12).std()
    
    # rolling statistics plot
    plt.figure()
    original = plt.plot(timeseries, color='blue', label='Original')
    mean = plt.plot(rolling_mean, color='red', label='Rolling Mean')
    std = plt.plot(rolling_std, color='black', label='Rolling Std')
    plt.legend(loc='best')
    plt.title('Rolling Mean & Standard Deviation')
    plt.show(block=False)
    
    # Dickey–Fuller test:
    result = adfuller(timeseries['Close'])
    print('ADF Statistic: {}'.format(result[0]))
    print('p-value: {}'.format(result[1]))
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t{}: {}'.format(key, value))
        
rolling_mean = closedf_arima_log.rolling(window=12).mean()
closedf_arima_log_minus_mean = closedf_arima_log - rolling_mean
closedf_arima_log_minus_mean.dropna(inplace=True)
get_stationarity(closedf_arima_log_minus_mean)

rolling_mean_exp_decay = closedf_arima_log.ewm(halflife=12, min_periods=0, adjust=True).mean()
closedf_arima_log_exp_decay = closedf_arima_log - rolling_mean_exp_decay
closedf_arima_log_exp_decay.dropna(inplace=True)
get_stationarity(closedf_arima_log_exp_decay)

closedf_arima_log_shift = closedf_arima_log - closedf_arima_log.shift()
closedf_arima_log_shift.dropna(inplace=True)
get_stationarity(closedf_arima_log_shift)
#%%
decomposition = seasonal_decompose(closedf_arima_log, period=1)
model = ARIMA(closedf_arima_log, order=(4,1,2))
results = model.fit()
plt.figure()
plt.plot(closedf_arima_log)
plt.plot(results.fittedvalues, color='red')
plt.show()

#%%
predictions_ARIMA_diff = pd.Series(results.fittedvalues, copy=True)
predictions_ARIMA_diff_cumsum = predictions_ARIMA_diff.cumsum()
predictions_ARIMA_log = pd.Series(closedf_arima_log['Close'].iloc[0], index=closedf_arima_log.index)
predictions_ARIMA_log = predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum, fill_value=0)
predictions_ARIMA = np.exp(predictions_ARIMA_log)
plt.plot(closedf_arima, 'r')
plt.plot(predictions_ARIMA, 'b')
plt.ylim([0, 100e5])
#%%
'''
#%%
'''
model=Sequential()
model.add(LSTM(time_step, input_shape=(None,1), activation="relu"))
model.add(Dense(1))
model.compile(loss="mean_squared_error", optimizer="adam")
'''
'''
# Define the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(time_step, 1)))
model.add(Dropout(0.2))
model.add(LSTM(50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50))
model.add(Dropout(0.2))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
'''
'''
model = Sequential()
model.add(Conv1D(50, 3, activation='relu', input_shape=(time_step, 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(20, activation='relu'))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')
'''
model = Sequential()
model.add(LSTM(256, return_sequences=True, input_shape=(time_step, 1)))
model.add(LSTM(256))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
history = model.fit(X_train,y_train,validation_data=(X_test,y_test), epochs=50, batch_size=50, verbose=1, shuffle=False)
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(loss))
plt.figure()
plt.plot(epochs, loss, 'r', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()

### Lets Do the prediction and check performance metrics
train_predict=model.predict(X_train)

#%%
temp = []
for i in range(19):
    temp.append(list(X_train[-1, i+1, :])[0])
temp.append(list(train_predict[-1])[0])
temp = np.array(temp)
temp = np.reshape(temp, (1, len(temp), 1))
#%%
final_pred = []
final_pred.append(model.predict(temp)[0][0])
#%%
for i in range(50):
    temp1 = []
    for j in range(len(temp[0])):
        temp1.append(list(temp[:, j, :])[0][0])
    temp1.pop(0)
    temp1.append(final_pred[i])
    temp1 = np.array(temp1)
    temp1 = np.reshape(temp1, (1, len(temp1), 1))
    final_pred.append(model.predict(temp1, verbose=0)[0][0])
    print(str(int((i+1)/len(X_test)*100))+" %",end='.r')
    temp = temp1
#%%%
test_predict=model.predict(X_test)
final_pred = np.array(final_pred)
final_pred = np.reshape(final_pred, (len(final_pred), 1))
# Transform back to original form
train_predict1 = scaler.inverse_transform(train_predict)
final_pred1 = scaler.inverse_transform(final_pred)
test_predict1 = scaler.inverse_transform(test_predict)
original_ytrain = scaler.inverse_transform(y_train.reshape(-1,1)) 
original_ytest = scaler.inverse_transform(y_test.reshape(-1,1)) 

# shift train predictions for plotting
look_back=time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict1
print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = final_pred1
print("Test predicted data: ", testPredictPlot.shape)
#%%
names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


plotdf = pd.DataFrame({'date': datac['Datetime'],
                       'original_close': datac['Close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['date'], y=[plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':'Stock price','date': 'Date'})
fig.update_layout(title_text='Comparision between original close price vs predicted close price',
                  plot_bgcolor='white', font_size=15, font_color='black', legend_title_text='Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.show()