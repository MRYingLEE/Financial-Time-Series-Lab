
# Financial Lab or Live Cheat Sheet


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

## Single Name


```python
# ! PIP install pandas_datareader 
# To install data reader, which is to read external data by web API
```


```python
def loaddata(ticker):
    '''To load daily data of 1 stock'''
    from pandas_datareader.data import DataReader
    from datetime import datetime, date # Date & time functionality
    start = date(2015, 1, 1) # Default: Jan 1, 2010
    end = date(2019, 1, 1)
    # ticker = 'GOOG'
    data_source = 'iex'
    stock_data = DataReader(ticker, data_source, start, end)
    return stock_data
```


```python
def loaddatas(tickers):
    '''To load daily data of some stocks'''
    ss=[]
    for ticker in tickers:
        stock=loaddata(ticker)
        stock[ticker]=stock['close'].pct_change()
        ss.append(stock[ticker])
 
    
    datas=pd.concat(ss, names=tickers,axis=1)
    
    return datas     
        
```


```python
one_stock=loaddata('GOOG')
one_stock['return']=one_stock['close'].pct_change()
one_stock.set_index(pd.to_datetime(one_stock.index), inplace=True)
one_stock.to_csv('goog.csv') # We save data file, so we can use it later for continue demo
one_stock.to_excel('goog.xlsx')
one_stock.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 1006 entries, 2015-01-02 to 2018-12-31
    Data columns (total 6 columns):
    open      1006 non-null float64
    high      1006 non-null float64
    low       1006 non-null float64
    close     1006 non-null float64
    volume    1006 non-null int64
    return    1005 non-null float64
    dtypes: float64(5), int64(1)
    memory usage: 55.0 KB
    

### Read Data from CSV (Optional)


```python
df_csv = pd.read_csv('goog.csv',na_values='n/a',parse_dates=['date'])
df_csv.set_index(df_csv['date'],inplace=True) # inplace parameter is very important
df_csv.drop(['date'],axis=1, inplace=True)
df_csv.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 1006 entries, 2015-01-02 to 2018-12-31
    Data columns (total 6 columns):
    open      1006 non-null float64
    high      1006 non-null float64
    low       1006 non-null float64
    close     1006 non-null float64
    volume    1006 non-null int64
    return    1005 non-null float64
    dtypes: float64(5), int64(1)
    memory usage: 55.0 KB
    

### Read Data from Excel


```python
df_excel = pd.read_excel('goog.xlsx', sheet_name='Sheet1',na_values='n/a',index_col=0)
df_excel.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>return</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>529.01</td>
      <td>531.270</td>
      <td>524.10</td>
      <td>524.81</td>
      <td>1446662</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>523.26</td>
      <td>524.330</td>
      <td>513.06</td>
      <td>513.87</td>
      <td>2054238</td>
      <td>-0.020846</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>515.00</td>
      <td>516.175</td>
      <td>501.05</td>
      <td>501.96</td>
      <td>2891950</td>
      <td>-0.023177</td>
    </tr>
    <tr>
      <th>2015-01-07</th>
      <td>507.00</td>
      <td>507.244</td>
      <td>499.65</td>
      <td>501.10</td>
      <td>2059366</td>
      <td>-0.001713</td>
    </tr>
    <tr>
      <th>2015-01-08</th>
      <td>497.99</td>
      <td>503.480</td>
      <td>491.00</td>
      <td>502.68</td>
      <td>3344395</td>
      <td>0.003153</td>
    </tr>
  </tbody>
</table>
</div>



### To calculate daily return statistics feature


```python
one_stock['return'].plot(kind="hist", bins=75, density=False)
plt.show()
```


![png](Financial%20Lab_files/Financial%20Lab_12_0.png)



```python
print("Mean=", np.mean(one_stock["return"]))
print("Annualized Mean=", ((1+np.mean(one_stock['return'])**252-1)))
print("std=", np.std(one_stock['return']))
print("Annualized std=",np.std(one_stock['return'])*np.sqrt(252))

from scipy.stats import skew
print("skew=", skew(one_stock['return']))
from scipy.stats import kurtosis
print("kurtosis=", kurtosis(one_stock['return']))

# The null hypothesis of the Shapiro-Wilk test is that the data are normally distributed.
from scipy import stats
p_value = stats.shapiro(one_stock['return'])[1]
if p_value <= 0.05:
    print("Null hypothesis of normality is rejected.")
else:
    print("Null hypothesis of normality is accepted.")
```

    Mean= 0.000789390761141073
    Annualized Mean= 0.0
    std= 0.015102839920493941
    Annualized std= 0.2397501511226729
    skew= nan
    kurtosis= nan
    Null hypothesis of normality is accepted.
    

### Downsampling: To turn daily data to weekly data


```python
resampler=one_stock.resample('W')
resampler_open=resampler['open'].first()
resampler_high=resampler['high'].max()
resampler_low=resampler['low'].min()
resampler_close=resampler['close'].last()
resampler_volume=resampler['volume'].sum()
resampler_series=[resampler_open,resampler_high,resampler_low,resampler_close,resampler_volume]
```


```python
# Method 1: column based
one_stock_weekly=pd.concat(resampler_series,axis=1)
one_stock_weekly["return"] = one_stock_weekly["close"].pct_change()
one_stock_weekly.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>return</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-04</th>
      <td>529.01</td>
      <td>531.27</td>
      <td>524.100</td>
      <td>524.81</td>
      <td>1446662</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-11</th>
      <td>523.26</td>
      <td>524.33</td>
      <td>491.000</td>
      <td>496.17</td>
      <td>12415664</td>
      <td>-0.054572</td>
    </tr>
    <tr>
      <th>2015-01-18</th>
      <td>494.94</td>
      <td>508.19</td>
      <td>487.560</td>
      <td>508.08</td>
      <td>11919169</td>
      <td>0.024004</td>
    </tr>
    <tr>
      <th>2015-01-25</th>
      <td>511.00</td>
      <td>542.17</td>
      <td>506.016</td>
      <td>539.95</td>
      <td>9433420</td>
      <td>0.062726</td>
    </tr>
    <tr>
      <th>2015-02-01</th>
      <td>538.53</td>
      <td>539.87</td>
      <td>501.200</td>
      <td>534.52</td>
      <td>14883499</td>
      <td>-0.010056</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Method 2: Row based and transpose
one_stock_weekly=pd.DataFrame(resampler_series)
one_stock_weekly=one_stock_weekly.T # important
one_stock_weekly["return"] = one_stock_weekly["close"].pct_change()
one_stock_weekly.head() 
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>return</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-04</th>
      <td>529.01</td>
      <td>531.27</td>
      <td>524.100</td>
      <td>524.81</td>
      <td>1446662.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-11</th>
      <td>523.26</td>
      <td>524.33</td>
      <td>491.000</td>
      <td>496.17</td>
      <td>12415664.0</td>
      <td>-0.054572</td>
    </tr>
    <tr>
      <th>2015-01-18</th>
      <td>494.94</td>
      <td>508.19</td>
      <td>487.560</td>
      <td>508.08</td>
      <td>11919169.0</td>
      <td>0.024004</td>
    </tr>
    <tr>
      <th>2015-01-25</th>
      <td>511.00</td>
      <td>542.17</td>
      <td>506.016</td>
      <td>539.95</td>
      <td>9433420.0</td>
      <td>0.062726</td>
    </tr>
    <tr>
      <th>2015-02-01</th>
      <td>538.53</td>
      <td>539.87</td>
      <td>501.200</td>
      <td>534.52</td>
      <td>14883499.0</td>
      <td>-0.010056</td>
    </tr>
  </tbody>
</table>
</div>



### Rolling: To move average daily data
This is totally different from the weekly data


```python
one_stock_ma5=df_csv.rolling(window=5).mean()
one_stock_ma5["return"] = one_stock_ma5["close"].pct_change()
one_stock_ma5.dropna(inplace=True)
one_stock_ma5.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>open</th>
      <th>high</th>
      <th>low</th>
      <th>close</th>
      <th>volume</th>
      <th>return</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-09</th>
      <td>509.602</td>
      <td>511.2298</td>
      <td>499.910</td>
      <td>503.156</td>
      <td>2483132.8</td>
      <td>-0.011256</td>
    </tr>
    <tr>
      <th>2015-01-12</th>
      <td>503.938</td>
      <td>505.5590</td>
      <td>494.810</td>
      <td>498.892</td>
      <td>2536374.4</td>
      <td>-0.008475</td>
    </tr>
    <tr>
      <th>2015-01-13</th>
      <td>500.706</td>
      <td>502.9200</td>
      <td>493.078</td>
      <td>497.736</td>
      <td>2431121.8</td>
      <td>-0.002317</td>
    </tr>
    <tr>
      <th>2015-01-14</th>
      <td>498.236</td>
      <td>502.1172</td>
      <td>491.748</td>
      <td>497.690</td>
      <td>2465176.2</td>
      <td>-0.000092</td>
    </tr>
    <tr>
      <th>2015-01-15</th>
      <td>499.752</td>
      <td>502.5572</td>
      <td>493.100</td>
      <td>497.512</td>
      <td>2338568.2</td>
      <td>-0.000358</td>
    </tr>
  </tbody>
</table>
</div>




```python
# multiple line plot
plt.plot('close', data=one_stock_ma5,  color='red', linewidth=1, label="MA5")
plt.plot('close', data=one_stock_weekly, color='blue', linewidth=1, label="Weekly")

plt.legend()
```




    <matplotlib.legend.Legend at 0x2204b2a6278>




![png](Financial%20Lab_files/Financial%20Lab_20_1.png)


## Portfolio


```python
### To Prepare Data
one_portfolio=loaddatas(['GOOG','AAPL','FB','IBM'])
one_portfolio.set_index(pd.to_datetime(one_portfolio.index), inplace=True)
one_portfolio.info()
```

    <class 'pandas.core.frame.DataFrame'>
    DatetimeIndex: 1006 entries, 2015-01-02 to 2018-12-31
    Data columns (total 4 columns):
    GOOG    1005 non-null float64
    AAPL    1005 non-null float64
    FB      1005 non-null float64
    IBM     1005 non-null float64
    dtypes: float64(4)
    memory usage: 39.3 KB
    


```python
one_portfolio.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOOG</th>
      <th>AAPL</th>
      <th>FB</th>
      <th>IBM</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>-0.020846</td>
      <td>-0.028172</td>
      <td>-0.016061</td>
      <td>-0.015735</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>-0.023177</td>
      <td>0.000094</td>
      <td>-0.013473</td>
      <td>-0.021566</td>
    </tr>
    <tr>
      <th>2015-01-07</th>
      <td>-0.001713</td>
      <td>0.014021</td>
      <td>0.000000</td>
      <td>-0.006536</td>
    </tr>
    <tr>
      <th>2015-01-08</th>
      <td>0.003153</td>
      <td>0.038423</td>
      <td>0.026592</td>
      <td>0.021735</td>
    </tr>
  </tbody>
</table>
</div>




```python
portfolio_weights = np.array([0.25, 0.35, 0.20, 0.20]) # We suppose this is a FIXED weight strategy
one_portfolio["Portfolio"] = one_portfolio.iloc[:,0:4].mul(portfolio_weights, axis=1).sum(axis=1)
# The daily return
one_portfolio["Portfolio_cum"] = ((1+one_portfolio["Portfolio"]).cumprod()-1) 
# The cumulative return from the beginning
one_portfolio.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>GOOG</th>
      <th>AAPL</th>
      <th>FB</th>
      <th>IBM</th>
      <th>Portfolio</th>
      <th>Portfolio_cum</th>
    </tr>
    <tr>
      <th>date</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2015-01-02</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>2015-01-05</th>
      <td>-0.020846</td>
      <td>-0.028172</td>
      <td>-0.016061</td>
      <td>-0.015735</td>
      <td>-0.021431</td>
      <td>-0.021431</td>
    </tr>
    <tr>
      <th>2015-01-06</th>
      <td>-0.023177</td>
      <td>0.000094</td>
      <td>-0.013473</td>
      <td>-0.021566</td>
      <td>-0.012769</td>
      <td>-0.033926</td>
    </tr>
    <tr>
      <th>2015-01-07</th>
      <td>-0.001713</td>
      <td>0.014021</td>
      <td>0.000000</td>
      <td>-0.006536</td>
      <td>0.003172</td>
      <td>-0.030862</td>
    </tr>
    <tr>
      <th>2015-01-08</th>
      <td>0.003153</td>
      <td>0.038423</td>
      <td>0.026592</td>
      <td>0.021735</td>
      <td>0.023902</td>
      <td>-0.007698</td>
    </tr>
  </tbody>
</table>
</div>



### Correlation Matrix


```python
correlation_matrix = one_portfolio.iloc[:,0:4].corr()
print(correlation_matrix)
```

              GOOG      AAPL        FB       IBM
    GOOG  1.000000  0.509856  0.607780  0.413908
    AAPL  0.509856  1.000000  0.455920  0.385336
    FB    0.607780  0.455920  1.000000  0.293542
    IBM   0.413908  0.385336  0.293542  1.000000
    
