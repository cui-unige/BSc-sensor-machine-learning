from __future__ import print_function
from sklearn import linear_model
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import random


# Adrien Chabert

# The purpose of this python code is to create some function that will be use to initialize some dataframe with some information.
# There is also some function that will help to find the best watering.
# This code is done for the fill "DataCeres.csv" and "DataDemeter.csv". If you want to use this code for other data file. There are
# some changment to do (especially in the function preparation()).
# Your data file must have : the first column is date,
# the second column is mean_moisture and the third is mean_temperature


class Traitement(object):
    def __init__(self):
        self.W = None

        # This function add the information about de variation of the moisture and the temperature in one iteration
    def ajoutData(self,df):
        df['moistureAdd'] = df.index
        df['temperatureAdd'] = df.index

        # size
        nligne = df.shape[0]
        nColumn = df.shape[1]
        tmp = np.linspace(0,nligne-2,nligne-1,dtype=int)
        df.iloc[tmp, 3] = pd.Series(np.append(df.iloc[tmp+1,1].to_numpy()-df.iloc[tmp,1].to_numpy(),0), index=df.index)
        df.iloc[tmp, 4] = pd.Series(np.append(df.iloc[tmp+1,2].to_numpy()-df.iloc[tmp,2].to_numpy(),0), index=df.index)

        #Ajout de la ligne
        df.iloc[nligne-1, 3] = 0
        df.iloc[nligne-1, 4] = 0

        return df

        # This fonction add the information about the number day the experience start
    def addDay(self,df, init):
        df['index'] = df.index
        nligne = df.shape[0]
        position = df.shape[1]-1
        for i in range(nligne):
            df.iloc[i,position] = (i-init)//48 + 1
        return df


    #This function add the information about the quantity that was used for watering and the time since the last watering
    # df is the dataframe. pot si the corresponding bac of the dataframe. 1 = demeter and 2 = ceres
    def arrosageHist(self,df,pot):
        df['Arrosage'] = df.index #This quantity is not zero when we watering. So it's always at 10.30 or 11.30
        df['TAfterArrosage'] = df.index #Time quantity
        df['ArrosageHist'] = df.index #how much have been watered at the last watering
        aro = [0,0,0,0,0,0]
        if ((pot != 1) & (pot != 2)):
            print("Ce bac est inconnu")
            return df
        if pot == 1:
            aro = [10,20,40,35,45,15]
        elif pot == 2:
            aro = [10,15,30,20,10,10]
        j = 0
        nligne = df.shape[0]
        df.iloc[:,5] = 0
        tmp = 0
        for i in range(nligne):
            # this condition help us to find when the watering have been done
            if i in df.loc[df['moistureAdd'] >= 1].index:
                # The watering depend on the day it was done
                if pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 3, 28,0,0,0):
                    df.iloc[int(i),5] = aro[0]
                elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 4, 18,0,0,0):
                    df.iloc[int(i),5] = aro[1]
                elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 5, 3,0,0,0):
                    df.iloc[int(i),5] = aro[2]
                elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 5, 23,0,0,0):
                    df.iloc[int(i),5] = aro[3]
                elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 5, 30,0,0,0):
                    df.iloc[int(i),5] = aro[4]
                else:
                    df.iloc[int(i),5] = aro[5]

                j = 0
                tmp = df.iloc[int(i),5]

            # add the information about "since how long it has been watered"
            df.iloc[int(i),6] = j*30
            # add the information "What was the last watering"
            df.iloc[i,7] = tmp
            j = j + 1
        return df



    # Eliminate the NaN value. There is Nan value when the raserberry pi was not working
    # df is the dataframe. i is "what is the number of ligne of the first watering"
    # the first watering correspond to start of a cycle
    def eliminateNaNValue(self,df,i):
        nligne = df.shape[0]
        while(i+48 < nligne):
            if df.iloc[range(i,i+48),:].isnull().values.any():
                df.drop(df.index[range(i,i+48)],axis = 0,inplace = True)
                nligne = df.shape[0]
            else:
                i = i + 48
        return df



    # find the best prediction of the watering quantity with a triple regressions in one day
    # regAro, redEva and regSta are the regression
    # start is the starting moisture, purpose is the moisture wanted,
    # hightemp and lowtemp are list of temperature during the day
    # iteration is the number of period to predict. One period is 30 minutes
    # limite is the nomber of minute for the separation between the phase of evaporation and stailization
    def prediction3(self,regAro, regEva, regSta,limite,start,highTemp,lowTemp,purpose,iteration):
        val = [0,5,10,15,20,25,30,35,40,45,50]
        resBest = 1000
        evalBest = []
        aroBest = 0
        for el in val:
            # calcul the prediction
            res = self.calcul(el,regAro, regEva, regSta,limite,start,highTemp,lowTemp,purpose,iteration)
            if abs(resBest-purpose) > abs(res[-1]-purpose):
                resBest = res[-1]
                evalBest = res
                aroBest = el

            #print(el, res[48])
        return evalBest, aroBest



    # Calcul the moisture on one day
    # regAro, redEva and regSta are the regression
    # start is the starting moisture, purpose is the moisture wanted,
    # hightemp and lowtemp are list of temperature during the day
    # iteration is the number of period to predict. One period is 30 minutes
    # limite is the nomber of minute for the separation between the phase of evaporation and stailization
    def calcul(self,aro,regAro,regEva,regSta,limite,start,highTemp,lowTemp,purpose,iteration):
        res = np.zeros(iteration+1)
        res[0] = start
        if (aro != 0):
            res[1] = res[0] + regAro.predict([[res[0],aro]])[0]
            reg = regEva
            for i in range(2,iteration+1):
                if i == limite//30:
                    reg = regSta
                if i <= 24:
                    res[i] = res[i-1] + reg.predict([[res[i-1],highTemp,i*30,aro]])[0]
                else:
                    res[i] = res[i-1] + reg.predict([[res[i-1],lowTemp,i*30,aro]])[0]
        #if there is no watering
        else:
            for i in range(1,iteration+1):
                if i <= 24:
                    res[i] = res[i-1] + regSta.predict([[res[i-1],highTemp,i*30,aro]])[0]
                else:
                    res[i] = res[i-1] + regSta.predict([[res[i-1],lowTemp,i*30,aro]])[0]
        return res[range(1,iteration+1)]



 # do the preparation of the day for the calcul. Read the file. Add variation of moisture and temperature
 # Add the time after watering and the watering quantity. Eliminate NaN value
 # init is the ligne where there is the first watering. It's the start of "day"
    def preparation(self, name, init):
        df = pd.read_csv(name)
        #Create other columns
        df = self.ajoutData(df)
        print("Preparation des donnees ...")
        pot = 1 # pot = 1 is Demeter
        if (name == "DataCeres.csv"):
            pot = 2 # pot = 2 is Ceres
        df = self.arrosageHist(df,pot)
        df = self.addDay(df,init)

        #Delete the data that are not inside a loop of one day
        df.drop(df.index[range(0,init)],axis = 0,inplace = True)

        # Delete the day with problem
        # This part is really depending on the data. We eliminate value that seem to be wrong
        if (name == "DataDemeter.csv"):
            df.drop(df.index[df['index'] == 76], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 75], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 15], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 102], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 96], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 97], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 98], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 41], axis = 0, inplace = True)
        if (name == "DataCeres.csv"):
            df.drop(df.index[df['index'] == 1], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 2], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 3], axis = 0, inplace = True)
            df.drop(df.index[df['index'] < 51], axis = 0, inplace = True) #Avant 5
            df.drop(df.index[df['index'] == 74], axis = 0, inplace = True)
            df.drop(df.index[df['index'] == 89], axis = 0, inplace = True)
            df.drop(df.index[df['TAfterArrosage'] > 1410], axis = 0, inplace = True)

        #Eliminate NaN value.
        df = self.eliminateNaNValue(df,init)
        return df



    # Separate train data and test data. Df is the dataFrame to separate and
    # njour is the number of day of test data.
    # It's return 2 database. One of the training data and one of test data.
    def partitionTest(self, df, njour):
        nligne = df.shape[0]
        jour = nligne //48
        incr = 0
        jourTest = random.sample(range(1,jour),k=njour)
        jourTest.sort()
        indTest = []
        for el in jourTest:
            indTest = np.append(indTest,range(int(48*(el-1)),int(48*el)))
            indTest = indTest.astype(int)
        dataTest = df.iloc[indTest,:].copy()
        df.drop(df.index[indTest],axis = 0,inplace = True)
        return df, dataTest
