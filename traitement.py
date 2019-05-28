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

    def arrosage(self,df):
        df['Arrosage'] = df.index
        df['TAfterArrosage'] = df.index

        j = 0
        nligne = df.shape[0]
        df.iloc[:,5] = 0
        for i in range(nligne):
            if i in df.loc[df['moistureAdd'] >= 3].index:
                if pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 3, 28,0,0,0):
                    df.iloc[int(i),5] = 10
                elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 4, 18,0,0,0):
                    df.iloc[int(i),5] = 15
                elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 5, 3,0,0,0):
                    df.iloc[int(i),5] = 30
                else:
                    df.iloc[int(i),5] = 20
                j = 0
            df.iloc[int(i),6] = j*30
            j = j + 1
        return df

    def arrosageHist(self,df,pot):
        df['Arrosage'] = df.index
        df['TAfterArrosage'] = df.index
        df['ArrosageHist'] = df.index
        aro = [0,0,0,0,0]
        if pot == 1:
            aro = [10,20,40,35,45]
        elif pot == 2:
            aro = [10,15,30,20,10]
        j = 0
        nligne = df.shape[0]
        df.iloc[:,5] = 0
        tmp = 0
        for i in range(nligne):
            if i in df.loc[df['moistureAdd'] >= 3].index:
                if pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 3, 28,0,0,0):
                    df.iloc[int(i),5] = aro[0]
                elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 4, 18,0,0,0):
                    df.iloc[int(i),5] = aro[1]
                elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 5, 3,0,0,0):
                    df.iloc[int(i),5] = aro[2]
                elif pd.to_datetime(df.iloc[i,0]) < datetime.datetime(2019, 5, 23,0,0,0):
                    df.iloc[int(i),5] = aro[3]
                else:
                    df.iloc[int(i),5] = aro[4]

                j = 0
                tmp = df.iloc[int(i),5]

            df.iloc[int(i),6] = j*30
            df.iloc[i,7] = tmp
            j = j + 1
        return df


    def eliminateNaNValue(self,df,i):
        nligne = df.shape[0]
        while(i+48 < nligne):
            if df.iloc[range(i,i+48),:].isnull().values.any():
                df.drop(df.index[range(i,i+48)],axis = 0,inplace = True)
                nligne = df.shape[0]
                print(i)
            else:
                i = i + 48
        return df

    def pure(self,df):
        dfPure = df.loc[df['TAfterArrosage'] <= 30 ]
        nligne = df.shape[0]
        i=0
        j= 47
        while(j < nligne):
            dfPure.iloc[i,2] = np.mean(df.iloc[range(j-47,j-24),2].values)
            dfPure.iloc[i+1,2] = np.mean(df.iloc[range(j-24,j),2].values)
            if (j + 48 <=nligne):
                dfPure.iloc[i+1,3] = dfPure.iloc[i+2,1]-dfPure.iloc[i+1,1]
            i = i +2
            j = j + 48
        return dfPure

    def prediction(self,reg,start,highTemp,lowTemp,purpose):
        val = [5,10,15,20,25,30,35,40]
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

    def prediction3(self,regAro, regEva, regSta,limite,start,highTemp,lowTemp,purpose):
        val = [5,10,15,20,25,30,35,40]
        resBest = 1000
        evalBest = []
        aroBest = 0
        for el in val:
            res = np.zeros(49)
            res[0] = start
            res[1] = res[0] + regAro.predict([[res[0],el]])[0]
            reg = regEva
            for i in range(2,49):
                if i == limite//30:
                    reg = regSta
                if i <= 24:
                    res[i] = res[i-1] + reg.predict([[res[i-1],highTemp,i*30]])[0]
                else:
                    res[i] = res[i-1] + reg.predict([[res[i-1],lowTemp,i*30]])[0]
            if abs(resBest-purpose) > abs(res[48]-purpose):
                resBest = res[48]
                evalBest = res
                aroBest = el
            print(el, res[48])
        return evalBest, aroBest

    def preparation(self, name, init):
        df = pd.read_csv(name)
        #Création des deux autres colonnes
        df = self.ajoutData(df)
        df = self.arrosageHist(df,1)
        df = self.eliminateNaNValue(df,init)
        #Delete the data that are not inside a loop of one day
        df.drop(df.index[range(0,init)],axis = 0,inplace = True)
        return df

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
