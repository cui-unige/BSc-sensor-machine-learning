from pandas import Series
from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np


def lecture(name):
    #series = read_csv(name, header=0, parse_dates=[0], index_col=[0])
    series = Series.from_csv(name, header=0, sep=',',index_col =0)
    # split dataset
    X = series.values
    print(X)
    #X = series['mean_moisture-percent']
    #print(X.loc['2019-04-01 22:00:00'])
    train, test = X[1:len(X)-50], X[len(X)-50:]
    return train, test

def AutoRegress(train, test):
    # train autoregression
    model = AR(train)
    model_fit = model.fit()
    window = model_fit.k_ar
    coef = model_fit.params
    # walk forward over time steps in test
    history = train[len(train)-window:]
    history = [history[i] for i in range(len(history))]
    predictions = list()
    for t in range(len(test)):
    	length = len(history)
    	lag = [history[i] for i in range(length-window,length)]
    	yhat = coef[0]
    	for d in range(window):
    		yhat += coef[d+1] * lag[window-d-1]
    	obs = test[t]
    	predictions.append(yhat)
    	history.append(obs)

    	print('predicted=%f, expected=%f' % (yhat, obs))
    error = mean_squared_error(test, predictions)
    print('Test MSE: %.3f' % error)
    return predictions, history[len(history)-50:]

# create a differenced series
def difference(dataset, interval):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval):
	return yhat + history[-interval]

def arima(data, interval):
    n = 10
    differenced = difference(data, interval)
    # fit model
    model = ARIMA(differenced, order=(n,0 ,1))
    model_fit = model.fit(disp=0)
    # make prediction
    # multi-step out-of-sample forecast
    start_index = len(differenced)
    end_index = start_index + n
    forecast = model_fit.predict(start=start_index, end=end_index)
        # invert the differenced forecast to something usable
    history = data
    day = 1
    for yhat in forecast:
    	inverted = inverse_difference(history, yhat, interval)
    	print('minute %d: %f' % (day, inverted))
    	np.append(history,inverted)
    	day += 1
    error = mean_squared_error(test[0:n], history[len(history)-n:])
    print('Test MSE: %.3f' % error)
    return history[len(history)-n:]

def Affichage(predictions, test):
    # plot
    pyplot.plot(test[:])
    pyplot.plot(predictions, color='red')
    pyplot.legend(('expected','predicted'))
    pyplot.title('ARIMA : Forecast of moisture on 10 iteration of 5 minutes')
    pyplot.show()

name = 'ceres-chirp-right500h.csv'
train, test = lecture(name)

#predictions,eval = AutoRegress(train,test)
#Affichage(predictions, eval)

interval = int(60*24/5) #nomber of 5 minute in one day
predictions = arima(train,interval)
Affichage(predictions, test[0:10])
