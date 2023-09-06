import copy
import pandas as pd
import matplotlib.pyplot as plt

def sumColumns(columns, df, newColumnName):
    df[newColumnName] = df[columns].sum(axis = 1)
    return df

def setTimeIndex(df):
    newDf = copy.deepcopy(df)
    newDf['time'] = pd.to_datetime(newDf['time'])
    newDf = newDf.set_index(newDf['time'])

    return newDf    

def buildSnowPlot(snowFilePath, baseDf, column):
    snowDf = pd.read_csv(snowFilePath)
    snowDf['time'] = snowDf['Unnamed: 0']
    snowDf['time'] = pd.to_datetime(snowDf['time'])
    snowDf = snowDf.set_index(snowDf['time'])
    snowDf.loc[snowDf['SNOWFALL'] > 1] = 1
    snow = snowDf['SNOWFALL']
    
    ACprod = baseDf['AC_pred'].resample('D').sum()
    Prod = baseDf[column].resample('D').sum()

    ratio = Prod/ACprod

    plt.plot(ratio, label = 'Prediction/Actual for {}'.format(column))
    plt.plot(snow)

    plt.xlabel('Date')

def addRatio(column1, column2, df):
    c1Prod = df[column1].resample('D').sum()
    c2Prod = df[column2].resample('D').sum()
    ratio = c1Prod/c2Prod
    ratio = ratio.resample('15T').ffill()

    df['{}/{}'.format(column1, column2)] = ratio

    return df

def partitionBySnow(ratio, column, df):
    cleanDays = df.loc[df[column] > ratio]
    snowDays = df.loc[df[column] < ratio]

    return cleanDays, snowDays

