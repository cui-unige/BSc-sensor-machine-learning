import csv
import math
import matplotlib.pyplot as plt
import numpy as np

# Adrien Chabert


def lecture(name):
    #Data place
    tabl = []

    nameTabl = [] #name des caractéristiques
    with open(name) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        ligne = 0
        for row in csv_reader:
            if ligne != 0:
                i = 0
                for el in row:

                    # !!!!!!!!!!!!!!!
                    #Il faut faire la transformation de la date
                    tabl[i].append(el)
                    i += 1
            else:
                for el in row:
                    tabl.append([])
                    nameTabl.append(el)

            ligne += 1
    array = np.array(tabl)
    print(nameTabl)
    return array, nameTabl


def partitionnement(data):
    nbData = (data.shape)[1]
    echant = data[:,0:nbData-8]
    test = data[:,nbData-8:-1]
    return echant, test

name = 'Data2learn.csv'
All, name = lecture(name)
#print(All[:,1])

train, test = partitionnement(All)
print(test)


# from pandas import Series
# from pandas import DataFrame
# from pandas import concat
# from matplotlib import pyplot
# from sklearn.metrics import mean_squared_error
# series = Series.from_csv('Data2learn.csv', header=0)
# # create lagged dataset
# values = DataFrame(series.values)
# dataframe = concat([values.shift(1), values], axis=1)
# dataframe.columns = ['t-1', 't+1']
# # split into train and test sets
# X = dataframe.values
# train, test = X[1:len(X)-50], X[len(X)-50:]
# train_X, train_y = train[:,0], train[:,1]
# test_X, test_y = test[:,0], test[:,1]
#
# # persistence model
# def model_persistence(x):
# 	return x
#
# # walk-forward validation
# predictions = list()
# for x in test_X:
# 	yhat = model_persistence(x)
# 	predictions.append(yhat)
# test_score = mean_squared_error(test_y, predictions)
# print('Test MSE: %.3f' % test_score)
# # plot predictions vs expected
# pyplot.plot(test_y)
# pyplot.plot(predictions, color='red')
# pyplot.show()



from pandas import Series
from matplotlib import pyplot
from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error
series = Series.from_csv('Data2learn.csv', header=0, sep=',',index_col =0)
# split dataset
X = series.values
train, test = X[1:len(X)-50], X[len(X)-50:]
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
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
