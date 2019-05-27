from sklearn import linear_model
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from traitement import Traitement

t= Traitement()


df = t.preparation('DataAll.csv', 37)
nligne = df.shape[0]
nColumn = df.shape[1]


#data test

dfTest = t.preparation('Data30AprilTo8Mai.csv', 17)
nligneTest = dfTest.shape[0]

#Separation of the data in 3 parts
dfArrosage = df.loc[df['TAfterArrosage'] == 0]
tmp = df.loc[df['TAfterArrosage'] > 0]
dfStabilisation = df.loc[df['TAfterArrosage'] > 450]
dfEvaporation = tmp.loc[df['TAfterArrosage'] <= 450]

#Linear Regression
reg = linear_model.LinearRegression()
reg.fit(df[['mean_moisture-percent','mean_temperature','Arrosage']],df.moistureAdd)
# regLinearAro = linear_model.LinearRegression()
# regLinearAro.fit(dfArrosage[['mean_moisture-percent','Arrosage']],dfArrosage.moistureAdd)
# regLinearAro.intercept_
# regLinearAro.coef_
#
# regLinearEva = linear_model.LinearRegression()
# regLinearEva.fit(dfEvaporation[['mean_moisture-percent','mean_temperature','TAfterArrosage','ArrosageHist']],dfEvaporation.moistureAdd)
# regLinearEva.intercept_
# regLinearEva.coef_
#
# regLinearSta = linear_model.LinearRegression()
# regLinearSta.fit(dfStabilisation[['mean_moisture-percent','mean_temperature','TAfterArrosage']],dfStabilisation.moistureAdd)
# regLinearSta.intercept_
# regLinearSta.coef_


#Test de nos données
# init = 0
# base = dfTest
# dfTest['resultSepare'] = dfTest.index
# StartHumidity = base.iloc[init,1]
# position = dfTest.shape[1]-1
# TestSize = nligneTest
# ErrAro = 0
# result = np.zeros(TestSize)
# ErrEva = 0
# nAro = 0
# for i in range(0,TestSize):
#     dfTest.iloc[int(i+init),position] = StartHumidity
#     if i%48 == 0:
#         nAro =+ 1
#         StartHumidity = dfTest.iloc[init+i,1]
#         result[i] = StartHumidity + regLinearAro.predict([[StartHumidity,base.iloc[init+i,5]]])[0]
#         ErrAro =+ dfTest.iloc[int(i),position] - base.iloc[i,1]
#     elif i%48 <= 15:
#         result[i] = StartHumidity + regLinearEva.predict([[StartHumidity,base.iloc[init+i,2],base.iloc[init+i,6],base.iloc[init+i,7]]])[0]
#         ErrEva =+ dfTest.iloc[int(i),position] - base.iloc[i,1]
#     else:
#         result[i] = StartHumidity + regLinearSta.predict([[StartHumidity,base.iloc[init+i,2],base.iloc[init+i,6]]])[0]
#         ErrEva =+ dfTest.iloc[int(i),position] - base.iloc[i,1]
#     StartHumidity = result[i]
#
# #Our erreur. 1 is for the watering and the other is for the period of evaporation
# ErrAro = ErrAro/nAro
# ErrEva = ErrEva/(TestSize-nAro)
#
# print("Moyenne Erreur d'Arrosage : ", ErrAro)
# print("Moyenne d'erreur evaporation : ", ErrEva)
# ErrSepare = np.mean(dfTest.iloc[:,1].values-dfTest.iloc[:,position].values)
# print("Moyenne d'erreur : ",ErrSepare)



#Creation de l'algorithme pour predire l'arrosage
[res, bestaro] = t.prediction(reg, 22,26,23,22)
print(bestaro)
print(res)
plt.plot(res)
plt.ylabel('moisture percent')
plt.xlabel('iteration')
plt.show()


print(t.pure(df))
