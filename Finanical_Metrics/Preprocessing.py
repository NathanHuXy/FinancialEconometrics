import pandas as pd
from scipy.stats.mstats import winsorize

def winsorize_panel(data:pd.DataFrame,label:str,down:str,up:float,date:str) ->pd.DataFrame:
    '''
    Return a winsorized array of specific label based on groupby objects.
    Notes
    -----
    This function exsecutes the date inplace, and return the winsored data simultaneously.

    '''
    data.sort_values(date,inplace=True)
    result = []
    for t,group in data.groupby(date):
        win = winsorize(group[label],limits=[down,up])
        result.append(win)
    result = [token for st in result for token in st]
    data[label+'_win']=result
    return data