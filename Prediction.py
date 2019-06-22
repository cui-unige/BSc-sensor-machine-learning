from sklearn import linear_model
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
from traitement import Traitement
import random
import sys

# print (sys.argv[0]) #Affiche monfichier.py
# start = float(sys.argv[1])
# end = float(sys.argv[2])


def BestWater(nDays, startMoist, purposeMoist, listHigh, listlow, regLinearAro, regLinearEva, regLinearSta):
    result = [startMoist]
    arrosage = []
    iter = 48

    for d in range(nDays):
        purposeDaily = result[-1] + (purposeMoist - result[-1])/(nDays-d)
        [res, bestaro] = t.prediction3(regLinearAro, regLinearEva, regLinearSta, 240, result[int(0+iter*d)],listHigh[int(d)], listlow[int(d)],purposeDaily, iter)
        result = [*result, *res]
        arrosage.append(bestaro)
    if (result[-1] < (purposeMoist-1)):
        print("Pour atteindre cette humidité, il faudra arrosager 2 fois par jour. ")
        result = [startMoist]
        arrosage = []
        iter = 24
        for d in range(nDays*2):
            temp = listlow
            if (d%2 == 0):
                temp = listHigh
            purposeDaily = result[-1] + (purposeMoist - result[-1])/(nDays*2-d)
            [res, bestaro] = t.prediction3(regLinearAro, regLinearEva, regLinearSta, 240, result[int(0+iter*d)],temp[int(d//2)], temp[int(d//2)],purposeDaily, iter)
            result = [*result, *res]
            arrosage.append(bestaro)
    return result, arrosage



#Preparation of the data
t= Traitement()
print("Import de la database")
df = t.preparation('DataDemeter.csv', 6)
nligne = df.shape[0]
nColumn = df.shape[1]



#Separation of the data in 3 parts
limite = 240
dfArrosage = df.loc[df['TAfterArrosage'] == 0]
tmp = df.loc[df['TAfterArrosage'] > 0]
dfStabilisation = df.loc[df['TAfterArrosage'] > limite]
dfEvaporation = tmp.loc[df['TAfterArrosage'] <= limite]
print("Calcul de la regression ...")
#Linear Regression
reg = linear_model.LinearRegression()
reg.fit(df[['mean_moisture-percent','mean_temperature','Arrosage']],df.moistureAdd)

# Regression on the 3 parts
regLinearAro = linear_model.LinearRegression()
regLinearAro.fit(dfArrosage[['mean_moisture-percent','Arrosage']],dfArrosage.moistureAdd)

regLinearEva = linear_model.LinearRegression()
regLinearEva.fit(dfEvaporation[['mean_moisture-percent','mean_temperature','TAfterArrosage','ArrosageHist']],dfEvaporation.moistureAdd)

regLinearSta = linear_model.LinearRegression()
regLinearSta.fit(dfStabilisation[['mean_moisture-percent','mean_temperature','TAfterArrosage','ArrosageHist']],dfStabilisation.moistureAdd)

#Creation de l'algorithme pour predire l'arrosage
# Avec une simple regression :
#[res, bestaro] = t.prediction(reg, 22,26,23,22)

# we collect the temperature and the moisture
print("\nVous désirez prédire la quantité d'eau que vous devrez arroser chaque jour pour atteindre un certain niveau d'humidité ? Notre algorithme est fait pour ça ! \nPour y arriver, nous avons besoin de quelques paramètres.\n")
nDay = int(input("Sur combien de jour ? : "))
start = float(input("\nHumidité actuel ? : "))
end = float(input("Humidité souhaité ? : "))

listHighTemp = np.zeros(nDay)
listLowTemp = np.zeros(nDay)
# Avec la separation en 3 de la journee:
for d in range(nDay):
    print("")
    print(" Pour le jour numero : ", d+1)
    listHighTemp[d] = float(input("temperature journée ? : "))
    listLowTemp[d] = float(input("temperature nuit ? : "))


[result, arrosage] = BestWater(nDay, start, end, listHighTemp, listLowTemp, regLinearAro, regLinearEva, regLinearSta)
print("\n Meilleure arrosage : ", arrosage)
print("Himditité final prédit : ", result[-1])

plt.plot(result)
plt.ylabel('moisture percent')
plt.xlabel('iteration')
plt.grid(True)
plt.title('Prediction de l humidité avec l arrosage proposé')
plt.show()
