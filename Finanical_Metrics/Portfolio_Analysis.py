import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS
from linearmodels import FamaMacBeth

def portfolio_analysis_uni_variable(data:pd.DataFrame, label:str, excess_ret:str, date:str, n:int,is_weighted:bool,weight:str)->pd.DataFrame:
    if is_weighted==True:

        ##分组
        data.sort_values(date, inplace=True)
        codes = []
        for t, group in data.groupby(date):
            cats = pd.qcut(group[label], n, labels=range(n))
            codes.append(np.array(cats))

        codes = [token for st in codes for token in st]
        data['codes'] = codes

        ##计算加权平均值
        month_list = []
        code_list = []
        weightlist = []
        for (month, code), excess_return in data.groupby([date, 'codes']):
            weightlist.append(np.average(excess_return[excess_ret]
                                         , weights=excess_return[weight]))
            month_list.append(month)
            code_list.append(code)
        temp = pd.DataFrame([month_list, code_list, weightlist]
                            , index=[date, 'codes', 'weight_mean']
                            ).T

        ##整理成所需要的格式
        group_name_list = []
        for i in range(n):
            group_name_list.append('group' + str(i + 1))
        cross_section_average = temp.pivot_table('weight_mean', index=date, columns='codes')
        cross_section_average = pd.DataFrame(cross_section_average.values, index=cross_section_average.index
                                             , columns=group_name_list)
        cross_section_average['group' + str(n) + '-' + 'group1'] = cross_section_average['group' + str(n)] - \
                                                                   cross_section_average['group1']

        return cross_section_average
    elif is_weighted==False:
        data.sort_values(date, inplace=True)
        ##分组
        codes = []
        for t, group in data.groupby(date):
            cats = pd.qcut(group[label], n, labels=range(n))
            codes.append(np.array(cats))

        codes = [token for st in codes for token in st]
        data['codes'] = codes

        group_name_list = []
        for i in range(n):
            group_name_list.append('group' + str(i + 1))
        grouped = data[excess_ret].groupby([data[date], data['codes']])
        cross_section_average = pd.DataFrame(grouped.mean()).unstack()
        cross_section_average = pd.DataFrame(cross_section_average.values
                                             , index=cross_section_average.index
                                             , columns=group_name_list
                                             )
        cross_section_average['group' + str(n) + '-' + 'group1'] = cross_section_average['group' + str(n)] - \
                                                                   cross_section_average['group1']
        return cross_section_average


def portfolio_analysis_double_variables(label_one:str, label_two:str, data:pd.DataFrame, rnext, date:str,is_weighted:bool, weight:str)->pd.DataFrame:
    if is_weighted==True:
        ###对每一个时间截面依据第一个分类指标分组分组
        codes = []
        data.sort_values(date, inplace=True)
        for t, group in data.groupby(date):
            cats = pd.qcut(group[label_one], 5, labels=[0, 1, 2, 3, 4])
            codes.append(np.array(cats))
        codes = [token for st in codes for token in st]
        data['codes_' + label_one] = codes

        ###对每一个时间截面依据第二个分类指标分组
        codes = []
        data.sort_values([date, 'codes_' + label_one], inplace=True)
        for (t, temp), group in data.groupby([date, 'codes_' + label_one]):
            cats = pd.qcut(group[label_two], 5, labels=[str(temp) + '_' + str(0)
                , str(temp) + '_' + str(1)
                , str(temp) + '_' + str(2)
                , str(temp) + '_' + str(3)
                , str(temp) + '_' + str(4)])
            codes.append(np.array(cats))
        codes = [token for st in codes for token in st]
        data['codes_' + label_one + '_' + label_two] = codes

        ##计算加权平均值
        month_list = []
        code_list = []
        weightlist = []
        for (month, code), excess_return in data.groupby([date, 'codes_' + label_one + '_' + label_two]):
            weightlist.append(np.average(excess_return[rnext]
                                         , weights=excess_return[weight]))
            month_list.append(month)
            code_list.append(code)
        temp = pd.DataFrame([month_list, code_list, weightlist]
                            , index=[date, 'codes_' + label_one + '_' + label_two, 'weight_mean']
                            ).T

        ##整理成所需要的格式
        columns_list = []
        for i in range(5):
            for j in range(5):
                columns_list.append(str(i) + '_' + str(j))
        cross_section_average = temp.pivot_table('weight_mean', index=date, columns='codes_' + label_one + '_' + label_two)
        cross_section_average = pd.DataFrame(cross_section_average.values, index=cross_section_average.index
                                             , columns=columns_list)

        ###计算多空组合的收益率之差
        cross_section_average['0_4_0'] = cross_section_average['0_4'] - cross_section_average['0_0']
        cross_section_average['1_4_0'] = cross_section_average['1_4'] - cross_section_average['1_0']
        cross_section_average['2_4_0'] = cross_section_average['2_4'] - cross_section_average['2_0']
        cross_section_average['3_4_0'] = cross_section_average['3_4'] - cross_section_average['3_0']
        cross_section_average['4_4_0'] = cross_section_average['4_4'] - cross_section_average['4_0']

        return cross_section_average
    elif is_weighted==False:
        def double_analysis(label_one, label_two, data, rnext, date):

            '''
            label_one:第一个分组的变量的字段名称
            label_two:第二个分组的变量的字段名称
            data:整合后的数据（分组变量1，分组变量2，股票代码，下一期超额收益率，时间）
            rnext：下一期超额收益率的字段名称
            date：时间的字段名称
            '''

            ###对每一个时间截面依据第一个分类指标分组分组
            codes = []
            data.sort_values([date], inplace=True)
            for t, group in data.groupby(date):
                cats = pd.qcut(group[label_one], 5, labels=[0, 1, 2, 3, 4])
                codes.append(np.array(cats))
            codes = [token for st in codes for token in st]
            data['codes_' + label_one] = codes

            ###对每一个时间截面依据第二个分类指标分组
            codes = []
            data.sort_values([date, 'codes_' + label_one], inplace=True)
            for (t, temp), group in data.groupby([date, 'codes_' + label_one]):
                cats = pd.qcut(group[label_two], 5, labels=[str(temp) + '_' + str(0)
                    , str(temp) + '_' + str(1)
                    , str(temp) + '_' + str(2)
                    , str(temp) + '_' + str(3)
                    , str(temp) + '_' + str(4)])
                codes.append(np.array(cats))
            codes = [token for st in codes for token in st]
            data['codes_' + label_one + '_' + label_two] = codes

            ###求两次分组后的组合的平均收益率
            grouped = data[rnext].groupby([data[date], data['codes_' + label_one + '_' + label_two]])
            cross_section_average = pd.DataFrame(grouped.mean()).unstack()

            ###整理成想要的格式
            columns_list = []
            for i in range(5):
                for j in range(5):
                    columns_list.append(str(i) + '_' + str(j))

            cross_section_average = pd.DataFrame(cross_section_average.values
                                                 , index=cross_section_average.index
                                                 , columns=columns_list
                                                 )

            ###计算多空组合的收益率之差
            cross_section_average['0_4_0'] = cross_section_average['0_4'] - cross_section_average['0_0']
            cross_section_average['1_4_0'] = cross_section_average['1_4'] - cross_section_average['1_0']
            cross_section_average['2_4_0'] = cross_section_average['2_4'] - cross_section_average['2_0']
            cross_section_average['3_4_0'] = cross_section_average['3_4'] - cross_section_average['3_0']
            cross_section_average['4_4_0'] = cross_section_average['4_4'] - cross_section_average['4_0']

            return cross_section_average

def calculate_beta_of_common_factor(data,window,x,y,symbol,date):
    # window表示滚动窗口的长度
    # 排序
    data.sort_values([symbol,date],inplace=True)
    # 用于储存每一个symbol的滚动回归的结果
    beta_list = []
    #利用groupby对每个symbol进行滚动回归处理
    for symbol,group in data.groupby(symbol):
        n = len(group)
        if n < window:
            np.full([1,5], np.nan)
            beta = pd.DataFrame({x+'_beta':np.full(n, np.nan),x+'_beta_const':np.full(n, np.nan)})[x+'_beta']
        else:
            X = group[x].values
            Y = group[y].values
            X = sm.add_constant(X)
            model = RollingOLS(Y,X,window=window,min_nobs=21)
            rres = model.fit(cov_type='HAC',cov_kwds={'maxlags':3})
            beta = pd.DataFrame(rres.params,columns=[x+'_beta_const',x+'_beta'])[x+'_beta']
        beta_list.append(beta)
    # 合并结果
    beta_result = pd.concat(beta_list,axis=0)
    return pd.DataFrame(beta_result)

def FM_regression(y,xs,panel,date,symbol,bandwidth):
    panel = panel[[y,date,symbol]+xs]
    panel.dropna(axis=0,inplace = True)
    ## panel是一个pandas面板数据(股票代码-日期面板)，且需设置multi-index，否则无法使用
    panel[date] = pd.to_datetime(panel[date]) # 日期需设置为datetime格式
    panel = panel.set_index([symbol, date]) # 设置multi-index
    formula = y + '~1'
    for x in xs:
        formula = formula + '+' + x
    mod = FamaMacBeth.from_formula(formula=formula, data=panel)
    ## `bandwidth`是Newey-West滞后阶数
    res = mod.fit(cov_type= 'kernel',debiased = False, bandwidth = bandwidth)
    print(res.summary)



# if __name__=='__main__':