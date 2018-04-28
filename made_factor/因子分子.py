from jaqs_fxdayu.util import dp
from jaqs.data.dataapi import DataApi



api = DataApi(addr='tcp://data.tushare.org:8910')
api.login("13662241013",
          'eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTc2NDQzMzg5MTIiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTM2NjIyNDEwMTMifQ.sVIzI5VLqq8fbZCW6yZZW0ClaCkcZpFqpiK944AHEow'
)

start = 20080101
end = 20180101
SH_id = dp.index_cons(api, "000300.SH", start, end)
SZ_id = dp.index_cons(api, "000905.SH", start, end)

stock_symbol = list(set(SH_id.symbol)|set(SZ_id.symbol))


factor_list = ['volume', 'pb', 'roe','close']
check_factor = ','.join(factor_list)
import jaqs_fxdayu
jaqs_fxdayu.patch_all()
from jaqs.data import DataView
from jaqs.data import RemoteDataService
from jaqs_fxdayu.data.dataservice import LocalDataService
import os
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from pandas import Series,DataFrame
dataview_folder = 'D:\data'
dv = DataView()
ds = LocalDataService(fp=dataview_folder)

dv_props = {'start_date': start, 'end_date': end, 'symbol':','.join(stock_symbol),
         'fields': check_factor,
         'freq': 1,
         "prepare_fields": True}

dv.init_from_config(dv_props, data_api=ds)
dv.prepare_data()
dv.add_field('sw1')

sw1 = dv.get_ts('sw1')
dict_classify = {'480000': '银行', '430000': '房地产', '460000': '休闲服务', '640000': '机械设备', '240000': '有色金属', '510000': '综合', '410000': '公用事业', '450000': '商业贸易', '730000': '通信', '330000': '家用电器', '720000': '传媒', '630000': '电气设备', '270000': '电子', '490000': '非银金融', '370000': '医药生物', '710000': '计算机', '280000': '汽车', '340000': '食品饮料', '220000': '化工', '210000': '采掘', '230000': '钢铁', '650000': '国防军工', '110000': '农林牧渔', '420000': '交通运输', '620000': '建筑装饰', '350000': '纺织服装', '610000': '建筑材料', '360000': '轻工制造'}

sw1_name = sw1.replace(dict_classify)
sw1_name.tail()

dv.add_field('close',ds)
dv.add_field('high',ds)
dv.add_field('low',ds)
dv.add_field('turnover',ds)
alpha1= dv.add_formula('alpha1', "Delta(close,200)/Delay(close,200)", is_quarterly=False, add_data=True)


alpha2 = dv.add_formula('alpha2',
               "Ts_Argmax(close,25)/25*100-Ts_Argmin(close,25)/25*100"
               , is_quarterly=False, add_data=True)

alpha3 = dv.add_formula('alpha3',
               "(high+low+close)*1/3*volume"
             , is_quarterly=False, add_data=True)

alpha4 = dv.add_formula('alpha4',
               "high/Delay(high,100)"
               , is_quarterly=False, add_data=True)
alpha5 = dv.add_formula('alpha5',
               "(close-Ts_Mean(close,20))/Ts_Mean(close,20)*100", is_quarterly=False, add_data=True)


alpha6 = dv.add_formula('alpha6',
               "((high-Ewma(close,1))-(low-Ewma(close,1)))/close"
               , is_quarterly=False, add_data=True)

alpha7 = dv.add_formula('alpha7',
               "Ts_Mean(Abs(close-Ts_Mean(close,10)),10)"
             , is_quarterly=False, add_data=True)

alpha8 = dv.add_formula('alpha8',
                                   "Ewma(Ewma(Ewma(close,10),10),10)/Delay(Ewma(Ewma(Ewma(close,10),10),10),1)-1"
                                   , is_quarterly=False, add_data=True)

alpha9 = dv.add_formula('alpha9',"Correlation(close,turnover,10)"
             , is_quarterly=False, add_data=True)

alpha10=dv.add_formula('alpha10',
                                  "2*(Ewma(close,5)-Ewma(close,10)-Ewma(Ewma(close,5)-Ewma(close,10),15))"
                                   , is_quarterly=False, add_data=True)

id_zz500 = dp.daily_index_cons(api, "000300.SH", start, end)
id_hs300 = dp.daily_index_cons(api, "000905.SH", start, end)

columns_500 = list(set(id_zz500.columns)-set(id_hs300.columns))

import pandas as pd
id_member = pd.concat([id_zz500[columns_500],id_hs300],axis=1)

mask = ~id_member
import numpy as np

# 定义可买卖条件——未停牌、未涨跌停
def limit_up_down():
    trade_status = dv.get_ts('trade_status').fillna(0)
    mask_sus = trade_status == 0
    # 涨停
    up_limit = dv.add_formula('up_limit', '(close - Delay(close, 1)) / Delay(close, 1) > 0.095', is_quarterly=False)
    # 跌停
    down_limit = dv.add_formula('down_limit', '(close - Delay(close, 1)) / Delay(close, 1) < -0.095', is_quarterly=False)
    can_enter = np.logical_and(up_limit < 1, ~mask_sus) # 未涨停未停牌
    can_exit = np.logical_and(down_limit < 1, ~mask_sus) # 未跌停未停牌
    return can_enter,can_exit

can_enter,can_exit = limit_up_down()

import numpy as np

alpha_signal =['alpha1','alpha2','alpha3','alpha4','alpha5','alpha6','alpha7','alpha8','alpha9','alpha10']
price = dv.get_ts('close_adj')
sw1 = sw1_name
enter = can_enter
exit =  can_exit
mask = mask

from jaqs_fxdayu.research.signaldigger.process import neutralize

neutralize_dict = {a: neutralize(factor_df = dv.get_ts(a), group = dv.get_ts("sw1")) for a in alpha_signal}

import matplotlib.pyplot as plt
from jaqs_fxdayu.research import SignalDigger
from jaqs_fxdayu.research.signaldigger import analysis

def cal_obj(signal, name, period, quantile):
#     price_bench = dv.data_benchmark
    obj = SignalDigger(output_folder="hs300/%s" % name,
                       output_format='pdf')
    obj.process_signal_before_analysis(signal,
                                   price=price,
                                   n_quantiles=quantile, period=period,
                                   mask=mask,
                                   group=sw1,
                                   can_enter = enter,
                                   can_exit = exit,
                                   commission = 0.0008
                                   )
    obj.create_full_report()
    return obj

def plot_pfm(signal, name, period=5, quantile=5):
    obj = cal_obj(signal, name, period, quantile)
    plt.show()
def signal_data(signal, name, period=5, quantile=5):
    obj = cal_obj(signal, name, period, quantile)
    return obj.signal_data

signals_dict = {a:signal_data(neutralize_dict[a], a, 20) for a in alpha_signal}
ic_pn = pd.Panel({a: analysis.ic_stats(signals_dict[a]) for a in signals_dict.keys()})
alpha_performance = round(ic_pn.minor_xs('return_ic'),2)
print(alpha_performance)
alpha_IR = alpha_performance.loc["Ann. IR"]
alpha_IC = alpha_performance.loc["IC Mean"]
good_alpha = alpha_IC[(abs(alpha_IC)>=0.03) & (abs(alpha_IR)>=0.25)]
good_alpha_dict = {g: float('%.2f' % good_alpha[g]) for g in good_alpha.index}



signal_dict = {alpha : signal_data(dv.get_ts(alpha), alpha, period=20, quantile=5) for alpha in good_alpha.index}
def ic_length(signal, days=750):
    return signal.loc[signal.index.levels[0][-days]:]

from jaqs.research.signaldigger import performance as pfm

performance_dict = {}
for alpha in good_alpha.index:
    ic = pfm.calc_signal_ic(ic_length(signal_dict[alpha]), by_group=True)
    mean_ic_by_group = pfm.mean_information_coefficient(ic, by_group=True)
    performance_dict[alpha] = round(mean_ic_by_group,2)

ic_industry = pd.Panel(performance_dict).minor_xs('ic')
High_IC_Industry = pd.DataFrame([ic_industry[ic_industry>=0.05][alpha].dropna(how='all') for alpha in good_alpha.index]).T

alpha1 = pd.Series({'name':'alpha1','data': ['close'] ,'IC':good_alpha_dict['alpha1'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':'Delta(close,200)/Delay(close,200)','parameter':[200],'description':'收盘价200天变化除以200天之前的收盘价','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha1'][indu]) for indu in High_IC_Industry['alpha1'].dropna().index}})
alpha2 = pd.Series({'name':'alpha2','data':  ['close','volume'] ,'IC':good_alpha_dict['alpha2'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':"Ts_Argmax(close,25)/25*100-Ts_Argmin(close,25)/25*100",'parameter':[25,100],'description':'自价格达到近期最高值和最低值依赖所经过的期间数','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha2'][indu]) for indu in High_IC_Industry['alpha2'].dropna().index}})
alpha3 = pd.Series({'name':'alpha3','data':  ['close','high','low','volume'] ,'IC':good_alpha_dict['alpha3'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':"(high+low+close)*1/3*volume",'parameter':[25,100],'description':'最高价最低价收盘价的均值乘以交易量','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha3'][indu]) for indu in High_IC_Industry['alpha3'].dropna().index}})
alpha4 = pd.Series({'name':'alpha4','data': ['volume','close','vwap'],'IC':good_alpha_dict['alpha4'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':"high/Delay(high,200)",'parameter':[100],'description':'最高价除以200天前最高价','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha4'][indu]) for indu in High_IC_Industry['alpha4'].dropna().index}})
alpha5 = pd.Series({'name':'alpha5','data': ['close'],'IC':good_alpha_dict['alpha5'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':"(close-Ts_Mean(close,20))/Ts_Mean(close,20)*100",'parameter':[20,100],'description':'收盘价和其20天移动平均的变化率','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha5'][indu]) for indu in High_IC_Industry['alpha5'].dropna().index}})
alpha6  = pd.Series({'name':'alpha6','data': ['close','low','high'],'IC':good_alpha_dict['alpha6'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':"((high-Ewma(close,1))-(low-Ewma(close,1)))/close",'parameter':[1],'description':'高价和收盘价一日指数移动平均差值与低价和收盘价一日指数移动平均的差值的差值除以收盘价','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha6'][indu]) for indu in High_IC_Industry['alpha6'].dropna().index}})
alpha7 = pd.Series({'name':'alpha7','data': ['close'],'IC':good_alpha_dict['alpha7'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':"Ts_Mean(Abs(close-Ts_Mean(close,10)),10)",'parameter':[10],'description':'收盘价差异的绝对值的均值','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha7'][indu]) for indu in High_IC_Industry['alpha7'].dropna().index}})
alpha8 = pd.Series({'name':'alpha8','data': ['close'],'IC':good_alpha_dict['alpha8'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':"Ewma(Ewma(Ewma(close,10),10),10)/Delay(Ewma(Ewma(Ewma(close,10),10),10),1)-1",'parameter':[100],'description':'先计算N日指数移动平均，之后再进行两次引动平均，计算其变化率','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha8'][indu]) for indu in High_IC_Industry['alpha8'].dropna().index}})
alpha9= pd.Series({'name':'alpha9','data': ['close,turnover'],'IC':good_alpha_dict['alpha9'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':"Correlation(close,turnover,10)",'parameter':[10],'description':'换手率和收盘价10日相关系数','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha9'][indu]) for indu in High_IC_Industry['alpha9'].dropna().index}})
alpha10= pd.Series({'name':'alpha10','data': ['close'],'IC':good_alpha_dict['alpha10'],'type':'价量类','market':'ZZ800','classify':'sw1','Formula':"2*(Ewma(close,5)-Ewma(close,10)-Ewma(Ewma(close,5)-Ewma(close,10),15))",'parameter':[5,10,15],'description':'由快的指数移动平均线减去慢的指数移动平均线得到快线DIF，再用2×（快线DIF-DIF的加权移动均线DEA）得到MACD柱','High_IC_Industry': {indu: float('%.2f' % High_IC_Industry['alpha10'][indu]) for indu in High_IC_Industry['alpha10'].dropna().index}})



save_excel = pd.concat([globals()[name] for name in High_IC_Industry.columns],axis=1,keys=High_IC_Industry.columns).T
save_excel
save_excel.to_excel('Finish_alpha.xlsx')