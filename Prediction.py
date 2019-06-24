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



# Adrien Chabert

# This python file is the implement of the logiciel that predict watering for demeter
# The client provide information about the actuel moisture, the moisture wanted, how many is the prediction and the temperature during this day.
# Then the logiciel give the optimal watering to reach this wanted moisture.




#The function Bestwater try to find the best watering.
# it can be done with 1,2 or 0 watering per day.
# regLinearAro is the model regression for the watering Time
# regLinearEva is the model regression for the evaporation Time
# regLinearSta is the model regression for the Stabilization Time
def BestWater(nDays, startMoist, purposeMoist, listHigh, listlow, regLinearAro, regLinearEva, regLinearSta):
    result = [startMoist]
    arrosage = []
    iter = 48 #48 iterations of 30 minutes is equal to one day. We try to find first the solution with 1 watering per day.

    for d in range(nDays):
        # To have homogenous watering during the day of prediction
        purposeDaily = result[-1] + (purposeMoist - result[-1])/(nDays-d)
        # prediction3 find the best watering
        [res, bestaro] = t.prediction3(regLinearAro, regLinearEva, regLinearSta, 240, result[int(0+iter*d)],listHigh[int(d)], listlow[int(d)],purposeDaily, iter)
        result = [*result, *res]
        arrosage.append(bestaro)

    # if one watering per day is not enough
    if (result[-1] < (purposeMoist-1)):
        print("Pour atteindre cette humidité, il faudra arrosager 2 fois par jour. ")
        result = [startMoist]
        arrosage = []
        iter = 24
        for d in range(nDays*2):
            temp = listlow
            if (d%2 == 0): # if it's the daytime.
                temp = listHigh # temp is the same because the prediction is one 12 hours.
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



#Separation of the data in 3 parts of the 3 regressions
limite = 240
dfArrosage = df.loc[df['TAfterArrosage'] == 0]
tmp = df.loc[df['TAfterArrosage'] > 0]
dfStabilisation = df.loc[df['TAfterArrosage'] > limite]
dfEvaporation = tmp.loc[df['TAfterArrosage'] <= limite]
print("Calcul de la regression ...")
# Regression on the 3 parts
# Wateing part
regLinearAro = linear_model.LinearRegression()
regLinearAro.fit(dfArrosage[['mean_moisture-percent','Arrosage']],dfArrosage.moistureAdd)
# Evaporation part
regLinearEva = linear_model.LinearRegression()
regLinearEva.fit(dfEvaporation[['mean_moisture-percent','mean_temperature','TAfterArrosage','ArrosageHist']],dfEvaporation.moistureAdd)
# Stabilization part
regLinearSta = linear_model.LinearRegression()
regLinearSta.fit(dfStabilisation[['mean_moisture-percent','mean_temperature','TAfterArrosage','ArrosageHist']],dfStabilisation.moistureAdd)



# we collect the temperature and the moisture
print("\nVous désirez prédire la quantité d'eau que vous devrez arroser dans le but d'atteindre un certain niveau d'humidité ? Notre algorithme est fait pour ça ! \nPour y arriver, nous avons besoin de quelques paramètres.\n")
nDay = int(input("Sur combien de jour ? : "))
start = float(input("Humidité actuelle ? : "))
end = float(input("Humidité souhaitée ? : "))

listHighTemp = np.zeros(nDay)
listLowTemp = np.zeros(nDay)
for d in range(nDay): # we collect the temperature
    print("")
    print("Pour le jour numéro : ", d+1)
    listHighTemp[d] = float(input("Temperature pendant la journée ? : "))
    listLowTemp[d] = float(input("Temperature pendant la nuit ? : "))


# try to find the best watering.
[result, arrosage] = BestWater(nDay, start, end, listHighTemp, listLowTemp, regLinearAro, regLinearEva, regLinearSta)
if (len(arrosage) == nDay):
    print("\nMeilleur arrosage journalier (1 arrosage par jour): ", arrosage)
    print("L'arrosage doit être fait à midi.")
elif (len(arrosage) == 2*nDay):
    print("\nMeilleur arrosage journalier (2 arrosage par jour): ", arrosage)
    print("L'arrosage doit être fait à midi et à minuit.")

print("Humditité finale prédite : ", result[-1])

# Print of the prediction of the moisture predict by the algorithme
plt.plot(result)
plt.ylabel('moisture percent')
plt.xlabel('iteration')
plt.grid(True)
plt.title('Prediction de l humidité avec l arrosage proposé')
plt.show()
