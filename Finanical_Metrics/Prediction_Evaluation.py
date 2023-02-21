import statsmodels.api as sm
import numpy as np
import pandas as pd

def DMW_test(df_predictions:pd.DataFrame,label_prediciton:str, HA_prediciton:str, excess_ret:str)->tuple:
    '''
    Returns result of Diebold & Mariano (1995) & West (1996) (DMW) test.

    Parameters
    ----------
    df_predictions :
        DataFrame of predictions and real dependent variables.
    label_prediction :
        Label(column name) of predictions to be compared with Historical Average(HA) predictions.
    HA_prediciton :
        Label(column name) of HA predictions.
    excess_ret :
        Label(column name) of excess return.

    Return
    ------
    Tuple of (t-values, p_values).

    Notes
    -----
    The t-value and p-value returned is the result of double-sides tests.
    '''
    DMW_result = df_predictions[[label_prediciton,HA_prediciton,excess_ret]].dropna()
    d_hat = (DMW_result[label_prediciton]-DMW_result['R_t+1'])**2-(DMW_result['R_t+1(forecast)_HA_24']-DMW_result['R_t+1'])**2
    res = sm.OLS(d_hat,np.ones([len(d_hat),1])).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    return res.tvalues,res.pvalues

def CW_test(df_predictions:pd.DataFrame,label1:str,label2:str, excess_ret):
    '''
    Return result of Clark & West(2007) test for nested forecasts.

    Parameters
    ----------
    df_predictions:
        DataFrame of predictions and real dependent variables.
    label1:
        The first label(column name) of predictions to be compared.
    label2:
        The second label(column name) of predictions to be compared.
    excess_ret :
        Label(column name) of excess return.
    Return:
        Result of Clark & West(2007) test for nested forecasts
    '''
    CW_result = df_predictions[[label1,label2,excess_ret]].dropna()
    d_hat = (CW_result[label1]-CW_result['R_t+1'])**2-(CW_result[label2]-CW_result['R_t+1'])**2
    f_hat = d_hat+(CW_result['R_t+1']-CW_result[label1])**2
    model = sm.OLS(f_hat,sm.add_constant(f_hat)['const']).fit(cov_type='HAC',cov_kwds={'maxlags':3})
    CW_PM_PR=model.tvalues[0]
    print('CW statistic【{}相比于{}】{}'.format(label1,label2,CW_PM_PR))
    return CW_PM_PR


if __name__=='__main__':
    data = pd.read_csv('/Users/yanyan/Documents/MyQuant/研究——前景因子的构建/时间序列分析/forefast_all_indexes.csv')
    data['TradingDate'] = pd.to_datetime(data['TradingDate'])
    CW_test(data,'R_t+1(forecast)_PCA_24','R_t+1(forecast)_HA_24','R_t+1')
