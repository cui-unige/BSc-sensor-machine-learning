from sklearn import linear_model
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from traitement import Traitement
import random


t= Traitement()


df = t.preparation('DataDemeterAll.csv', 6)
nligne = df.shape[0]
nColumn = df.shape[1]

#[df, dataTest] = t.partitionTest(df,7)
#print(df.shape[0])
#data test


#Separation of the data in 3 parts
limite = 450
dfArrosage = df.loc[df['TAfterArrosage'] == 0]
tmp = df.loc[df['TAfterArrosage'] > 0]
dfStabilisation = df.loc[df['TAfterArrosage'] > limite]
dfEvaporation = tmp.loc[df['TAfterArrosage'] <= limite]

#Linear Regression
reg = linear_model.LinearRegression()
reg.fit(df[['mean_moisture-percent','mean_temperature','Arrosage']],df.moistureAdd)

# Regression on the 3 parts
regLinearAro = linear_model.LinearRegression()
regLinearAro.fit(dfArrosage[['mean_moisture-percent','Arrosage']],dfArrosage.moistureAdd)
regLinearAro.intercept_
regLinearAro.coef_

regLinearEva = linear_model.LinearRegression()
regLinearEva.fit(dfEvaporation[['mean_moisture-percent','mean_temperature','TAfterArrosage']],dfEvaporation.moistureAdd)
regLinearEva.intercept_
regLinearEva.coef_

regLinearSta = linear_model.LinearRegression()
regLinearSta.fit(dfStabilisation[['mean_moisture-percent','mean_temperature','TAfterArrosage']],dfStabilisation.moistureAdd)
regLinearSta.intercept_
regLinearSta.coef_


#Creation de l'algorithme pour predire l'arrosage
# Avec une simple regression :
#[res, bestaro] = t.prediction(reg, 22,26,23,22)

# Avec la separation en 3 de la journee:
[res, bestaro] = t.prediction3(regLinearAro, regLinearEva, regLinearSta, limite, 21,26 ,23,23)

print("Meilleure arrosage : ", bestaro)
print("Evolution de l'humidite : ", res)
plt.plot(res)
plt.ylabel('moisture percent')
plt.xlabel('iteration')
plt.show()
