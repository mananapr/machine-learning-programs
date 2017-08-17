import pandas as pd
import quandl, math
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
import numpy as np

df = quandl.get('WIKI/GOOGL')

df = df[['Adj. Open', 'Adj. High', 'Adj. Low', 'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = ( (df['Adj. High'] - df['Adj. Close']) / (df['Adj. Close']) ) * 100.0
df['PCT_change'] = ( (df['Adj. Close'] - df['Adj. Open']) / (df['Adj. Open']) ) * 100
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

forecast_col = 'Adj. Close'
forecast_out = 5

df['label'] = df[forecast_col].shift(-forecast_out)

x = np.array(df.drop(['label'],1))
x_after = x[-forecast_out:]
x = x[:-forecast_out]
y = np.array(df['label']) 

#x = preprocessing.scale(x)
df.fillna(-999999, inplace=True)
x_train, x_test, y_train, y_test = train_test_split(x,y[:-forecast_out],test_size=0.20)

clf = LinearRegression()
clf.fit(x_train, y_train)

accuracy = clf.score(x_test, y_test)
forecast_set = clf.predict(x_after)
print(forecast_set,accuracy)
