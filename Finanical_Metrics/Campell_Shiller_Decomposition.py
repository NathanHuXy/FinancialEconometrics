import pandas as pd
import numpy as np
import statsmodels.api as sm

class Campbell_Shiller_Decomposition:
    def __init__(self,Use_HS300:bool):
        if Use_HS300==True:
            print('Using data of HS300 currently...')
            HS300_data = pd.read_csv('./Data/HS300_data.csv')
            HS300_data.rename(columns={
                '交易日期': 'TradingDate',
                '市盈率(PE)': 'PE',
                '股息率': 'DP'
            }, inplace=True)
            HS300_data['TradingDate'] = pd.to_datetime(HS300_data['TradingDate'])
            HS300_data.set_index('TradingDate', inplace=True)
            self.HS300_data_monthly = HS300_data.resample('M').mean()
        elif Use_HS300==False:
            print('Loading received market_data...')
#             to be completed in future version...

    def Decompose(self,index_data:pd.DataFrame,index_to_analyse:list):
        decomposing_data = pd.merge(index_data,self.HS300_data_monthly,right_index=True,left_index=True,how='outer')
        decomposing_data['DP_t+1'] = decomposing_data['DP'].shift(-1)
        decomposing_data['PE_t+1'] = decomposing_data['PE'].shift(-1)
        decomposing_data.dropna(inplace=True)

        print('被解释变量：DP_t+1')
        for label in index_to_analyse:
            model = sm.OLS(decomposing_data['DP_t+1'], sm.add_constant(decomposing_data[[label, 'DP']])).fit(
                cov_type='HAC', cov_kwds={'maxlags': 3})
            print("{}:: BETA t-value={}, p-value={}\n\t PHI t-value={}, p-value={}".format(label,
                                                                                           round(model.tvalues[1], 2),
                                                                                           model.pvalues[1],
                                                                                           round(model.tvalues[2], 2),
                                                                                           model.pvalues[2]))
            print('beta={}%, phi={}%'.format(round(model.params[1] * 100, 2), round(model.params[2] * 100, 2)))

        print()
        print('被解释变量：PE_t+1')
        for label in ['WA_PT', 'S_PT', 'CICSI', 'ISI', 'PLS_rolling', 'PCA']:
            model = sm.OLS(decomposing_data['PE_t+1'], sm.add_constant(decomposing_data[[label, 'DP']])).fit(
                cov_type='HAC', cov_kwds={'maxlags': 3})
            print("{}:: BETA t-value={}, p-value={}\n\t PHI t-value={}, p-value={}".format(label,
                                                                                           round(model.tvalues[1], 2),
                                                                                           model.pvalues[1],
                                                                                           round(model.tvalues[2], 2),
                                                                                           model.pvalues[2]))
            print('beta={}%, phi={}%'.format(round(model.params[1] * 100, 2), round(model.params[2] * 100, 2)))

if __name__=='__main__':
    Campbell_Shiller_Decomposition(Use_HS300=True)
