from jaqs_fxdayu.util import dp
from jaqs_fxdayu.data.dataapi import DataApi



api = DataApi(addr='tcp://data.tushare.org:8910')
api.login("13662241013",
          'eyJhbGciOiJIUzI1NiJ9.eyJjcmVhdGVfdGltZSI6IjE1MTc2NDQzMzg5MTIiLCJpc3MiOiJhdXRoMCIsImlkIjoiMTM2NjIyNDEwMTMifQ.sVIzI5VLqq8fbZCW6yZZW0ClaCkcZpFqpiK944AHEow'
)

start = 20100104
end = 20180425
SH_id = dp.index_cons(api, "000300.SH", start, end)
SZ_id = dp.index_cons(api, "000905.SH", start, end)

stock_symbol = list(set(SH_id.symbol)|set(SZ_id.symbol))


factor_list = ['volume', 'pb', 'roe','close']
check_factor = ','.join(factor_list)
import jaqs_fxdayu
jaqs_fxdayu.patch_all()
from jaqs_fxdayu.data import DataView
from jaqs_fxdayu.data import RemoteDataService
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
dv.add_field('turnover_ratio',ds)
dv.add_field('price_div_dps',ds)
dv.add_field('oper_rev',ds)
dv.add_field('roa',ds)
dv.add_field('total_share',ds)
dv.add_field('pe_ttm',ds)
dv.add_field('roe',ds)
dv.add_field('roa',ds)
#alpha1= dv.add_formula('alpha1', "-1*Delta(close,200)/Delay(close,200)", is_quarterly=False, add_data=True)

dv.add_formula('DP01',"close-Delay(Ts_Mean(close,20),11)",is_quarterly=False,add_data=True)
alpha1=dv.add_formula('alpha1','-Ts_Mean(DP01,6)',is_quarterly=False,add_data=True)
alpha2 = dv.add_formula('alpha2',
               "-Log(total_share*close)"
               , is_quarterly=False, add_data=True)

#alpha3 = dv.add_formula('alpha3',
#               "roa"
 #            , is_quarterly=False, add_data=True)

#alpha4 = dv.add_formula('alpha4',
#               "roe"
#               , is_quarterly=False, add_data=True)
#alpha5 = dv.add_formula('alpha5',
#               "-1*price_div_dps", is_quarterly=False, add_data=True)


alpha3 = dv.add_formula('alpha3',
               "1/pe_ttm"
               , is_quarterly=True, add_data=True)

#alpha7 = dv.add_formula('alpha7',
 #              "-1*Ts_Mean(Abs(close-Ts_Mean(close,10)),10)"
  #           , is_quarterly=False, add_data=True)

alpha4 = dv.add_formula('alpha4',
                                   "Ta('ATR',0,close,low,high,volume,20)" , is_quarterly=False, add_data=True)

alpha5 = dv.add_formula('alpha5',"-Correlation(close,turnover,10)"
             , is_quarterly=False, add_data=True)

alpha6=dv.add_formula('alpha6',
                                  "-Ts_Mean(turnover_ratio,25)"
                                   , is_quarterly=False, add_data=True)
alpha7=dv.add_formula('alpha7',"-(close-Ewma(close,12))/Ewma(close,12)*100",is_quarterly=False,add_data=True)


id_zz500 = dp.daily_index_cons(api, "000300.SH", start, end)
id_hs300 = dp.daily_index_cons(api, "000905.SH", start, end)

columns_500 = list(set(id_zz500.columns)-set(id_hs300.columns))

import pandas as pd
id_member = pd.concat([id_zz500[columns_500],id_hs300],axis=1)

mask = ~id_member
volume=dv.get_ts('volume')
mask=mask.reindex(index=volume.index)

mask.columns=volume.columns
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

from jaqs_fxdayu.research import SignalDigger
from jaqs_fxdayu.research.signaldigger import performance as pfm
from jaqs_fxdayu.research.signaldigger import multi_factor
obj = SignalDigger()
signal_data = dict()
ic = dict()

for signal in ["alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7"]:
    signal_data[signal] = dict()
    ic[signal] = dict()
    period=20
    obj.process_signal_before_analysis(signal=dv.get_ts(signal),
                                         price=dv.get_ts("close_adj"),
                                           high=dv.get_ts("high_adj"), # 可为空
                                           low=dv.get_ts("low_adj"),# 可为空
                                           n_quantiles=5,# quantile分类数
                                           mask=mask,# 过滤条件
                                           can_enter=can_enter,# 是否能进场
                                           can_exit=can_exit,# 是否能出场
                                           period=period,# 持有期
                                           benchmark_price=dv.data_benchmark, # 基准价格 可不传入，持有期收益（return）计算为绝对收益
                                           commission = 0.0008,
                                           )
    signal_data[signal][period] = obj.signal_data
    ic[signal][period] = pfm.calc_signal_ic(obj.signal_data)
import pandas as pd

ic_mean_table = pd.DataFrame(data=np.nan,columns=[20],index=["alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7"])
ic_std_table = pd.DataFrame(data=np.nan,columns=[20],index=["alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7"])
ir_table = pd.DataFrame(data=np.nan,columns=[20],index=["alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7"])



for signal in ["alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7"]:
    period=20
    ic_mean_table.loc[signal, period] = ic[signal][period].mean().values[0]
    ic_std_table.loc[signal, period] = ic[signal][period].std().values[0]
    ir_table.loc[signal, period] = ic[signal][period].mean().values[0] / ic[signal][period].std().values[0]

print(ic_mean_table)
print(ic_std_table)
print(ir_table)

import matplotlib
ic_mean_table.plot(kind="barh",xerr=ic_std_table,figsize=(15,5))
ir_table.plot(kind="barh",figsize=(15,5))
from jaqs_fxdayu.research.signaldigger import process

factor_dict = dict()
id_zz800 = dp.daily_index_cons(api, "000906.SH", start, end)

import pandas as pd
id_zz800 = dp.daily_index_cons(api, "000906.SH", start, end)
index_member = pd.concat([id_zz800],axis=1)

dv.add_field('float_mv')
for name in ["alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7"]:
    signal = dv.get_ts(name)
    process.winsorize(factor_df=signal, alpha=0.05, index_member=index_member)  # 去极值
    #signal = process.standardize(signal, index_member)  # z-score标准化 保留排序信息和分布信息
    #     signal = process.rank_standardize(signal,index_member) #因子在截面排序并归一化到0-1(只保留排序信息)
         # 行业市值中性化
    signal = process.neutralize(factor_df=signal,
                                    group=dv.get_ts("sw1"),# 行业分类标准
                                    float_mv = dv.get_ts("float_mv"), #流通市值 可为None 则不进行市值中性化
                                    index_member=index_member,# 是否只处理时只考虑指数成份股
                                    )
    factor_dict[name] = signal

new_factors = multi_factor.orthogonalize(factors_dict=factor_dict,
                           standardize_type="rank",#输入因子标准化方法，有"rank"（排序标准化）,"z_score"(z-score标准化)两种（"rank"/"z_score"）
                           winsorization=False,#是否对输入因子去极值
                           index_member=index_member) #　是否只处理指数成分股

props = {
    'price':dv.get_ts("close_adj"),
    'high':dv.get_ts("high_adj"), # 可为空
    'low':dv.get_ts("low_adj"),# 可为空
    'ret_type': 'return',#可选参数还有upside_ret/downside_ret 则组合因子将以优化潜在上行、下行空间为目标
    'benchmark_price': dv.data_benchmark,  # 为空计算的是绝对收益　不为空计算相对收益
    'period': 20, # 30天的持有期
    'mask': mask,
    'can_enter': can_enter,
    'can_exit': can_exit,
    'forward': True,
    'commission': 0.0008,
    "covariance_type": "shrink",  # 协方差矩阵估算方法 还可以为"simple"
    "rollback_period": 120}  # 滚动窗口天数

comb_factors = dict()
for method in ["equal_weight","ic_weight","ir_weight","max_IR","max_IC"]:
    comb_factors[method] = multi_factor.combine_factors(factor_dict,
                                                        standardize_type="rank",
                                                        winsorization=False,
                                                        weighted_method=method,
                                                        props=props)
    print(method)
    print(comb_factors[method].dropna(how="all").head())

period = 20

ic_20 = multi_factor.get_factors_ic_df(comb_factors,
                                           price=dv.get_ts("close_adj"),
                                           high=dv.get_ts("high_adj"),  # 可为空
                                           low=dv.get_ts("low_adj"),  # 可为空
                                           n_quantiles=5,  # quantile分类数
                                           mask=mask,  # 过滤条件
                                           can_enter=can_enter,  # 是否能进场
                                           can_exit=can_exit,  # 是否能出场
                                           period=period,  # 持有期
                                           benchmark_price=dv.data_benchmark,  # 基准价格 可不传入，持有期收益（return）计算为绝对收益
                                           commission=0.0008,
                                           )

ic_df=pd.DataFrame(data=np.nan,columns=["alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7"],index=ic_20.index)
for singal in ["alpha1","alpha2","alpha3","alpha4","alpha5","alpha6","alpha7"]:
    ic_df[singal]=ic[singal][20]
ic_20 = pd.concat([ic_20, -1 * ic_df], axis=1)


ic_20_mean = dict()
ic_20_std = dict()
ir_20 = dict()


for name in ic_20.columns:
    ic_20_mean[name]=ic_20[name].loc[20170101:].mean()
    ic_20_std[name]=ic_20[name].loc[20170101:].std()
    ir_20[name] = ic_20_mean[name]/ic_20_std[name]
print(ic_20_mean)
print(ir_20)

import datetime

trade_date = pd.Series(ic_20.index)
trade_date = trade_date.apply(lambda x: datetime.datetime.strptime(str(x), '%Y%m%d'))
ic_20.index = trade_date

pd.Series(ic_20_mean).plot(kind="barh",xerr=pd.Series(ic_20_std),figsize=(15,5))

ic_20[["equal_weight","ic_weight","ir_weight","max_IR","max_IC"]].plot(kind="line",figsize=(15,5),)
ic_20.loc[datetime.date(2017,1,3):,][["equal_weight","ic_weight","ir_weight","max_IR","max_IC"]].plot(kind="line",figsize=(15,5),)

import matplotlib.pyplot as plt
from jaqs_fxdayu.research.signaldigger.analysis import analysis
from jaqs_fxdayu.research import SignalDigger

obj = SignalDigger()
obj.process_signal_before_analysis(signal=comb_factors["ic_weight"],
                                   price=dv.get_ts("close_adj"),
                                   high=dv.get_ts("high_adj"), # 可为空
                                   low=dv.get_ts("low_adj"),# 可为空
                                   n_quantiles=5,# quantile分类数
                                   mask=mask,# 过滤条件
                                   can_enter=can_enter,# 是否能进场
                                   can_exit=can_exit,# 是否能出场
                                   period=20,# 持有期
                                   benchmark_price=dv.data_benchmark, # 基准价格 可不传入，持有期收益（return）计算为绝对收益
                                   commission = 0.0008,
                                   )
obj.create_full_report()
plt.show()

print(analysis(obj.signal_data,is_event=False,period=20))


excel_data = obj.signal_data[obj.signal_data['quantile']==5]["quantile"].unstack().replace(np.nan, 0).replace(5, 1)
print (excel_data.head())
excel_data.to_excel('./ic_weight_quantile_5.xlsx')

obj = SignalDigger()
obj.process_signal_before_analysis(signal=comb_factors["ir_weight"],
                                   price=dv.get_ts("close_adj"),
                                   high=dv.get_ts("high_adj"), # 可为空
                                   low=dv.get_ts("low_adj"),# 可为空
                                   n_quantiles=5,# quantile分类数
                                   mask=mask,# 过滤条件
                                   can_enter=can_enter,# 是否能进场
                                   can_exit=can_exit,# 是否能出场
                                   period=20,# 持有期
                                   benchmark_price=dv.data_benchmark, # 基准价格 可不传入，持有期收益（return）计算为绝对收益
                                   commission = 0.0008,
                                   )
obj.create_full_report()
plt.show()

print(analysis(obj.signal_data,is_event=False,period=20))


excel_data = obj.signal_data[obj.signal_data['quantile']==5]["quantile"].unstack().replace(np.nan, 0).replace(5, 1)
print (excel_data.head())
excel_data.to_excel('./ir_weight_quantile_5.xlsx')