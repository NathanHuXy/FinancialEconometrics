import pandas as pd
import numpy as np
import statsmodels.api as sm

class Volatility_Risk:
    def __init__(self,Use_HS300:bool):
        if Use_HS300 == True:
            print('Using data of HS300 currently...')
            HS300_daily_r = pd.DataFrame()

            for path in [
                '../市场波动风险/HS300日收益率 2005-01-01 至 2008-04-30',
                '../市场波动风险/HS300日收益率 2008-05-01 至 2012-04-30',
                '../市场波动风险/HS300日收益率 2012-05-01 至 2016-04-30',
                '../市场波动风险/HS300日收益率 2016-05-01 至 2020-04-30',
                '../市场波动风险/HS300日收益率 2020-05-01 至 2023-01-06',
            ]:
                HS300_daily_r = pd.concat([HS300_daily_r, pd.read_csv(path + '/IDX_Idxtrd.csv')])

            HS300_daily_r.rename(columns={
                'Idxtrd01': 'TradingDate',
                'Idxtrd08': 'R_t'
            }, inplace=True)

            HS300_daily_r['TradingDate'] = pd.to_datetime(HS300_daily_r['TradingDate'])
            HS300_daily_r.set_index('TradingDate', inplace=True)
            self.HS300_daily_r = HS300_daily_r[['R_t']]

        elif Use_HS300==False:
            print('Loading received market_data...')
#             to be completed in future version...

    def Volatility_regression(self,index_data:pd.DataFrame,index_to_analyse:list):
        VAR = (self.HS300_daily_r ** 2).resample('M').sum()
        VAR.rename(columns={'R_t': 'SVAR_t'}, inplace=True)

        VAR['LVAR_t'] = np.log(np.sqrt(VAR['SVAR_t']))
        VAR['LVAR_t+1'] = VAR['LVAR_t'].shift(-1)
        VAR.dropna(inplace=True)
        VAR = pd.merge(index_data, VAR, right_index=True, left_index=True, how='inner')
        VAR.dropna(inplace=True)

        for label in index_to_analyse:
            model = sm.OLS(VAR['LVAR_t+1'], sm.add_constant(VAR[[label, 'LVAR_t']])).fit(cov_type='HAC',
                                                                                         cov_kwds={'maxlags': 3})

            print("{}:: BETA t-value={}, p-value={}\n\t PSI t-value={}, p-value={}".format(label,
                                                                                           round(model.tvalues[1], 2),
                                                                                           model.pvalues[1],
                                                                                           round(model.tvalues[2], 2),
                                                                                           model.pvalues[2]))
            print('beta={}%, psi={}%'.format(round(model.params[1] * 100, 2), round(model.params[2] * 100, 2)))
            print('R^2={}%'.format(round(model.rsquared_adj * 100, 2)))
            print()