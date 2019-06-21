from __future__ import print_function
from sklearn import linear_model
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime
import random





class Traitement(object):
    def __init__(self):
        self.W = None

        # This function add the information about de variation of the moisture and the temperature in one iteration
    def ajoutData(self,df):
        df['moistureAdd'] = df.index
        df['temperatureAdd'] = df.index

        # size
        dt = 30
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

        #This function add the information about the quantity that was used for watering and since how long the has been watering
    # def arrosage(self,df):
    #     df['Arrosage'] = df.index #This quantity is not zero when we watering. So it's always at 10.30 or 11.30
    #     df['TAfterArrosage'] = df.index
    #
    #     j = 0
    #     nligne = df.shape[0]
    #     df.iloc[:,5] = 0
    #     for i in range(nligne):
    #         if i in df.loc[df['moistureAdd'] >= 3].index:
    #             if pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 3, 28,0,0,0):
    #                 df.iloc[int(i),5] = 10
    #             elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 4, 18,0,0,0):
    #                 df.iloc[int(i),5] = 15
    #             elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 5, 3,0,0,0):
    #                 df.iloc[int(i),5] = 30
    #             else:
    #                 df.iloc[int(i),5] = 20
    #             j = 0
    #         df.iloc[int(i),6] = j*30
    #         j = j + 1
    #     return df

    #This function add the information about the quantity that was used for watering and since how long the has been watering
    def arrosageHist(self,df,pot):
        df['Arrosage'] = df.index #This quantity is not zero when we watering. So it's always at 10.30 or 11.30
        df['TAfterArrosage'] = df.index #Time quantity
        df['ArrosageHist'] = df.index #how much have been water at the last watering
        aro = [0,0,0,0,0]
        if pot == 1:
            aro = [10,20,40,35,45,15]
        elif pot == 2:
            aro = [10,15,30,20,10,10]
        j = 0
        nligne = df.shape[0]
        df.iloc[:,5] = 0
        tmp = 0
        for i in range(nligne):
            if i in df.loc[df['moistureAdd'] >= 1].index:
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

            df.iloc[int(i),6] = j*30
            df.iloc[i,7] = tmp
            j = j + 1
        return df

    # Eliminate the NaN value
    def eliminateNaNValue(self,df,i):
        nligne = df.shape[0]
        while(i+48 < nligne):
            if df.iloc[range(i,i+48),:].isnull().values.any():
                df.drop(df.index[range(i,i+48)],axis = 0,inplace = True)
                nligne = df.shape[0]
            else:
                i = i + 48
        return df

    # def pure(self,df):
    #     dfPure = df.loc[df['TAfterArrosage'] <= 30 ]
    #     nligne = df.shape[0]
    #     i=0
    #     j= 47
    #     while(j < nligne):
    #         dfPure.iloc[i,2] = np.mean(df.iloc[range(j-47,j-24),2].values)
    #         dfPure.iloc[i+1,2] = np.mean(df.iloc[range(j-24,j),2].values)
    #         if (j + 48 <=nligne):
    #             dfPure.iloc[i+1,3] = dfPure.iloc[i+2,1]-dfPure.iloc[i+1,1]
    #         i = i +2
    #         j = j + 48
    #     return dfPure

    # make the prediction of the watering quantity for a single regression in one day
    # reg = single regression, start is the starting moisture, purpose is the moisture wanted, hightemp and lowtemp are list of temperature during the day
    def prediction(self,reg,start,highTemp,lowTemp,purpose):
        val = [5,10,15,20,25,30,35,40,45,50]
        resBest = 1000
        evalBest = []
        aroBest = 0
        for el in val:
            res = np.zeros(49)
            res[0] = start
            res[1] = res[0] + reg.predict([[res[0],highTemp,el]])[0]
            for i in range(2,49):
                if i <= 24:
                    res[i] = res[i-1] + reg.predict([[res[i-1],highTemp,0]])[0]
                else:
                    res[i] = res[i-1] + reg.predict([[res[i-1],lowTemp,0]])[0]
            if abs(resBest-purpose) > abs(res[48]-purpose):
                resBest = res[48]
                evalBest = res
                aroBest = el
            print(el, res[48])
        return evalBest, aroBest

    # find the best prediction of the watering quantity with a triple regressions in one day
    # regAro, redEva and regSta are the regression
    # start is the starting moisture, purpose is the moisture wanted,
    # hightemp and lowtemp are list of temperature during the day
    # day is the number of day that must be predict
    # limite is the nomber of minute for the separation between the phase of evaporation and stailization
    def prediction3(self,regAro, regEva, regSta,limite,start,highTemp,lowTemp,purpose,day):
        val = [0,5,10,15,20,25,30,35,40,45,50]
        resBest = 1000
        evalBest = []
        aroBest = 0
        for el in val:
            res = self.calcul(el,regAro, regEva, regSta,limite,start,highTemp,lowTemp,purpose,day)
            if abs(resBest-purpose) > abs(res[48]-purpose):
                resBest = res[48]
                evalBest = res
                aroBest = el

            #print(el, res[48])
        return evalBest, aroBest

    # Calcul the moisture on one day
    # regAro, redEva and regSta are the regression
    # start is the starting moisture, purpose is the moisture wanted,
    # hightemp and lowtemp are list of temperature during the day
    # day is the number of day that must be predict
    # limite is the nomber of minute for the separation between the phase of evaporation and stailization
    def calcul(self,aro,regAro,regEva,regSta,limite,start,highTemp,lowTemp,purpose,day):
        res = np.zeros(49)
        res[0] = start
        if (aro != 0):
            res[1] = res[0] + regAro.predict([[res[0],aro,day]])[0]
            reg = regEva
            for i in range(2,49):
                if i == limite//30:
                    reg = regSta
                if i <= 24:
                    res[i] = res[i-1] + reg.predict([[res[i-1],highTemp,i*30,aro,day]])[0]
                else:
                    res[i] = res[i-1] + reg.predict([[res[i-1],lowTemp,i*30,aro,day]])[0]
        else:
            for i in range(1,49):
                if i <= 24:
                    res[i] = res[i-1] + regSta.predict([[res[i-1],highTemp,i*30,aro,day]])[0]
                else:
                    res[i] = res[i-1] + regSta.predict([[res[i-1],lowTemp,i*30,aro,day]])[0]
        return res

 # do the preparation of the day for the calcul. Read the file. Add variation of moisture and temperature
 # Add the time after watering and the watering quantity. Eliminate NaN value
    def preparation(self, name, init):
        df = pd.read_csv(name)
        #Create other columns
        df = self.ajoutData(df)
        print("Preparation des donnees ...")
        pot = 1
        if (name == "DataCeres.csv"):
            pot = 2
        df = self.arrosageHist(df,pot)
        df = self.addDay(df,init)

        #Delete the data that are not inside a loop of one day
        df.drop(df.index[range(0,init)],axis = 0,inplace = True)

        # Delete the day with problem
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
