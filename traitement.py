from __future__ import print_function
from sklearn import linear_model
import csv
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import datetime




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
        aro = [0,0,0,0]
        if pot == 1:
            aro = [10,20,40,35]
        elif pot == 2:
            aro = [10,15,30,20]
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
                else:
                    df.iloc[int(i),5] = aro[3]
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
