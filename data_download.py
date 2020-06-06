# -*- coding: utf-8 -*-
"""
@Author: jifeng
@File Create: 20200606
@Last Modify: 20200606
@Function: from tushare get stock data, generate array
"""

import tushare as ts
import pandas as pd
import time
import numpy as np
ts.set_token('33aeeaf3e4e6b1cac85a6035f0adf9fe0efb9e386d9c20ad0d3e4b81')
pro = ts.pro_api()

hs300 = list(set(pro.index_weight(index_code='399300.SZ', start_date='20160101', end_date='20170101')['con_code']))

roe_hs300 = pd.DataFrame()
eps_hs300 = pd.DataFrame()
for stock in hs300:
    data = pro.query('fina_indicator', ts_code=stock, start_date='20160101', end_date='20170801')
    roe_hs300[stock] = data['roe']
    eps_hs300[stock] = data['eps']
    time.sleep(1)
    
   
sort_stock_list = list((roe_hs300.loc[5].argsort()+eps_hs300.loc[5].argsort()).argsort().sort_values().index)
data = [[0 for i in range(18)] for j in range(18)]

for stock in sort_stock_list:
    df = pro.query('daily', ts_code=stock, start_date='20160101', end_date='20200101')
    df.to_csv(stock+'.csv')
dt = pd.DataFrame()
for stock in sort_stock_list:
    print(stock)
    df = pd.read_csv(stock+'.csv',index_col=2)
    dt[stock] = df['close']
    print(df.head(1))
t=0     
for i in range(36-1):
        for j in range(i+1):
            k = i-j
            if k<18 and k>=0 and j<18:
                #df = pro.query('daily', ts_code=sort_stock_list[t], start_date='20160101', end_date='20200101')
                #df_price = df['close']
                t = t+1
                data[j][k]=dt[sort_stock_list[t]]
