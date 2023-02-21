import pandas as pd
import numpy as np


class Asset_Allocation:
    def __init__(self, dataframe):
        '''
        R_t+1(forecast)_WA_PT_60	R_t+1(forecast)_S_PT_60	R_t+1(forecast)_HA_60	R_t+1	Nrr1	Nrrmtdt	rf_t+1	sigma^2_t+1
TradingDate
2010-05-31	-0.224684	-0.224456	-0.203463	-0.281483	NRI01	0.1856	0.1856	NaN
2010-06-30	-0.236131	-0.233128	-0.205169	-0.261392	NRI01	0.1856	0.1856	NaN
2010-07-31	-0.239498	-0.246769	-0.203360	-0.066298	NRI01	0.1856	0.1856	NaN
2010-08-31	-0.219293	-0.222865	-0.203907	-0.173630	NRI01	0.1856	0.1856	NaN
2010-09-30	-0.212279	-0.209651	-0.203532	-0.174447	NRI01	0.1856	0.1856	NaN
        '''
        self.data = dataframe.dropna()

    def calculate_CER_and_Sharpe(self, GAMMA, forecast_label, trade_fee):
        '''
        Parameters
        ----------
        GAMMA: 风险厌恶系数
        forecast_label: 预测收益率序列的列名称
        trade_fee: in percent
        '''
        self.data['w'] = (1 / GAMMA) * (self.data[forecast_label] / self.data['sigma^2_t+1']).shift(1)

        def scale(x):
            # if x<=1.5: return 0 ## 卖空限制 最多50%的杠杆
            # else: return x
            return x  # 无卖空限制

        self.data['w'] = self.data['w'].map(scale)

        self.data['R^p_{t+1}'] = self.data['w'] * (self.data['R_t+1'] - trade_fee) + self.data['rf_t+1']

        print("CER gain = {};\tSharpe Ratio = {}.".format(round(
            12 * (self.data['R^p_{t+1}'].dropna().mean() - 0.5 * GAMMA * self.data['R^p_{t+1}'].dropna().var()),
            2), round((self.data['R^p_{t+1}'] - self.data['rf_t+1']).dropna().mean() / (
            (self.data['R^p_{t+1}'] - self.data['rf_t+1']).dropna().std()), 2)))

        return round(12 * (self.data['R^p_{t+1}'].dropna().mean() - 0.5 * GAMMA * self.data['R^p_{t+1}'].dropna().var()),2), round((self.data['R^p_{t+1}'] - self.data['rf_t+1']).dropna().mean() / ((self.data['R^p_{t+1}'] - self.data['rf_t+1']).dropna().std()), 2)
