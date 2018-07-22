
# coding: utf-8

# In[7]:

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')
from matplotlib.pyplot import rcParams
rcParams['figure.figsize']=10,6


# In[8]:

dataset=pd.read_csv("AirPassengers.csv")
dataset['Month']=pd.to_datetime(dataset['Month'],infer_datetime_format=True)
indexedDataset=dataset.set_index(['Month'])



# In[9]:

from datetime import datetime
indexedDataset.head(5)


# In[10]:

plt.xlabel("Date")
plt.ylabel("number of air passengers")
plt.plot(indexedDataset)


# In[11]:

rolmean=indexedDataset.rolling(window=12).mean()
rolstd=indexedDataset.rolling(window=12).std()
print(rolmean,rolstd)


# In[12]:

orig=plt.plot(indexedDataset,color='blue',label='original')
mean=plt.plot(rolmean,color='red',label="Rolling mean")
std=plt.plot(rolstd,color='black',label="Rolling standard")
plt.legend(loc='best')
plt.title("Rolling mean & Rolling standard")
plt.show(block=False)


# In[13]:

from statsmodels.tsa.stattools import adfuller
print("result of dickey fuller test")
dftest=adfuller(indexedDataset['#Passengers'],autolag='AIC')
dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of observations Used'])
for key,value in dftest[4].items():
    dfoutput['criticalvalue(%s)'%key]=value
print(dfoutput)


# In[14]:

indexedDataset_logScale=np.log(indexedDataset)
plt.plot(indexedDataset_logScale)


# In[15]:

movingAverage=indexedDataset_logScale.rolling(window=12).mean()
movingSTD=indexedDataset_logScale.rolling(window=12).std()
plt.plot(indexedDataset_logScale)
plt.plot(movingAverage,color='red')


# In[16]:

datasetLogScaleMinusMovingAverage=indexedDataset_logScale-movingAverage
datasetLogScaleMinusMovingAverage.head(12)
datasetLogScaleMinusMovingAverage.dropna(inplace=True)
datasetLogScaleMinusMovingAverage.head(10)


# In[17]:

from statsmodels.tsa.stattools import adfuller
def test_stationary(timeseries):
    movingAverage=timeseries.rolling(window=12).mean()
    movingSTD=timeseries.rolling(window=12).std()
    orig=plt.plot(timeseries,color='blue',label='Original')
    mean=plt.plot(movingAverage,color='red',label='Rolling mean')
    std=plt.plot(movingSTD,color='black',label='Rolling std')
    plt.legend(loc='best')
    plt.title("Rolling mean & Rolling standard")
    plt.show(block=False)
    print("result of dicky fuller test")
    dftest=adfuller(timeseries['#Passengers'],autolag='AIC')
    dfoutput=pd.Series(dftest[0:4],index=['Test Statistic','p-value','#Lags Used','Number of observations Used'])
    for key,value in dftest[4].items():
        dfoutput['criticalvalue(%s)'%key]=value
    print(dfoutput)
    


# In[18]:

test_stationary(datasetLogScaleMinusMovingAverage)


# In[19]:

exponentialDecayWeightedAverage=indexedDataset_logScale.ewm(halflife=12,min_periods=0,adjust=True).mean()
plt.plot(indexedDataset_logScale)
plt.plot(exponentialDecayWeightedAverage,color='red')


# In[20]:

datasetLogScaleMinusMovingExponentialDecayAverage=indexedDataset_logScale-exponentialDecayWeightedAverage
test_stationary(datasetLogScaleMinusMovingExponentialDecayAverage)


# In[21]:

datasetLogDiffshifting=indexedDataset_logScale-indexedDataset_logScale.shift()
plt.plot(datasetLogDiffshifting)


# In[22]:

datasetLogDiffshifting.dropna(inplace=True)
test_stationary(datasetLogDiffshifting)


# In[23]:

from statsmodels.tsa.seasonal import seasonal_decompose
decomposition=seasonal_decompose(indexedDataset_logScale)
trend=decomposition.trend
seasonal=decomposition.seasonal
residual=decomposition.resid
plt.subplot(411)
plt.plot(indexedDataset_logScale,label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend,label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual,label="residual")
plt.legend(loc='best')
plt.tight_layout()


decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)


# In[24]:


decomposedLogData=residual
decomposedLogData.dropna(inplace=True)
test_stationary(decomposedLogData)


# In[25]:

from statsmodels.tsa.stattools import acf,pacf
lag_acf=acf(datasetLogDiffshifting,nlags=20)
lag_pacf=pacf(datasetLogDiffshifting,nlags=20,method='ols')
plt.subplot(121)
plt.plot(lag_acf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='--',color='gray')
plt.title("Autocorrelation function")
plt.subplot(122)
plt.plot(lag_pacf)
plt.axhline(y=0,linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='--',color='gray')
plt.axhline(y=1.96/np.sqrt(len(datasetLogDiffshifting)),linestyle='--',color='gray')
plt.title("Autocorrelation function")
plt.tight_layout()


# In[30]:

from statsmodels.tsa.arima_model import ARIMA
model=ARIMA(indexedDataset_logScale,order=(2,1,2))
results_ARIMA=model.fit(disp=-1)
plt.plot(datasetLogDiffshifting)
plt.plot(results_ARIMA.fittedvalues,color='red')
plt.title('RSS:%.4f'%sum((results_ARIMA.fittedvalues-datasetLogDiffshifting['#Passengers'])**2))
print('Plotting AR model')


# In[32]:

predictions_ARIMA_diff=pd.Series(results_ARIMA.fittedvalues,copy=True)
print(predictions_ARIMA_diff)


# In[34]:

predictions_ARIMA_diff_cumsum=predictions_ARIMA_diff.cumsum()
print(predictions_ARIMA_diff_cumsum.head())


# In[35]:

predictions_ARIMA_log=pd.Series(indexedDataset_logScale['#Passengers'].ix[0],index=indexedDataset_logScale.index)
predictions_ARIMA_log=predictions_ARIMA_log.add(predictions_ARIMA_diff_cumsum,fill_value=0)
predictions_ARIMA_log.head()


# In[36]:

predictions_ARIMA=np.exp(predictions_ARIMA_log)
plt.plot(indexedDataset)
plt.plot(predictions_ARIMA)


# In[38]:

results_ARIMA.plot_predict(1,264)


# In[39]:

results_ARIMA.forecast(steps=120)


# In[ ]:



